import os, sys, json, time, traceback, shutil, csv, smtplib, ssl, mimetypes, re
from pathlib import Path
from datetime import datetime, timedelta, date
from email.message import EmailMessage
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf

try:
    import serial
except Exception:
    serial = None

BASE_DIR = Path(__file__).resolve().parent
RUNTIME = BASE_DIR / "runtime_status"
LOG_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
RECIPES_DIR = BASE_DIR / "recipes"
CONFIG_PATH = BASE_DIR / "config_usb.json"
EMAIL_CONFIG_PATH = BASE_DIR / "config_email.json"
HEARTBEAT_PATH = RUNTIME / "heartbeat.json"
LAST_RESULT_PATH = RUNTIME / "last_result.json"
SUMMARY_PATH = RUNTIME / "summary.json"
COMMAND_PATH = RUNTIME / "command.json"
ACK_PATH = RUNTIME / "ack.json"
IND_LOG_PATH = LOG_DIR / "industrial_log.jsonl"
CSV_LOG_PATH = LOG_DIR / "inspection_log.csv"
TRACE_LOG_PATH = LOG_DIR / "inspection_trace_log.csv"
MES_XML_DIR = LOG_DIR / "mes_xml"
CURRENT_CONTEXT_PATH = RUNTIME / "current_serial.json"
AUTO_REPORT_STATE_PATH = RUNTIME / "auto_report_state.json"
CORE_VERSION = "SVC_USB_v17_2_refresh500"
CORE_STARTED_AT_EPOCH = time.time()
OFFICIAL_CLASSES = ["OK", "NG_DESALINHADO", "NG_DANIFICADO"]

for p in [RUNTIME, LOG_DIR, REPORTS_DIR, MODELS_DIR, MES_XML_DIR]:
    p.mkdir(exist_ok=True, parents=True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def epoch_ms():
    """Timestamp de parede em ms para medir eventos entre app e core."""
    return round(time.time() * 1000.0, 3)


def atomic_write_json(path: Path, payload: dict, retries: int = 20, delay: float = 0.03):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    data = json.dumps(payload, indent=2, ensure_ascii=False)
    last = None
    for _ in range(retries):
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(str(tmp), str(path))
            return True
        except PermissionError as e:
            last = e
            time.sleep(delay)
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last if last else RuntimeError("atomic_write_json failed")


def read_json(path: Path, default=None):
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return default
    return default


def log_event(event: str, **kw):
    row = {"ts": now(), "event": event, **kw}
    with open(IND_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_config():
    cfg = read_json(CONFIG_PATH, {}) or {}
    cfg.setdefault("camera_index", 0)
    cfg.setdefault("camera_backend", "auto")
    cfg.setdefault("camera_width", 1280)
    cfg.setdefault("camera_height", 720)
    cfg.setdefault("roi_x0", 0.0)
    cfg.setdefault("roi_y0", 0.0)
    cfg.setdefault("roi_x1", 1.0)
    cfg.setdefault("roi_y1", 1.0)
    cfg.setdefault("img_size", 224)
    cfg.setdefault("confidence_threshold", 0.5)
    cfg.setdefault("captures_dir", "captures_usb")
    cfg.setdefault("dataset_dir", "dataset_usb_live_capture")
    cfg.setdefault("save_ng_images", True)
    cfg.setdefault("save_all_captures", False)
    cfg.setdefault("save_raw_and_roi", True)
    cfg.setdefault("auto_reconnect_camera", True)
    cfg.setdefault("auto_trigger_enabled", False)
    cfg.setdefault("auto_trigger_interval_s", 2.0)
    cfg.setdefault("retention_days", 30)
    cfg["class_mode"] = "3class"
    cfg["dataset_classes"] = OFFICIAL_CLASSES
    cfg.setdefault("serial_enabled", False)
    cfg.setdefault("serial_port", "COM1")
    cfg.setdefault("serial_baud", 115200)
    cfg.setdefault("trigger_mode", "edge_0to1")
    cfg.setdefault("sensor_settle_ms", 180)
    cfg.setdefault("serial_stable_ms", 80)
    cfg.setdefault("serial_rearm_ms", 600)
    cfg.setdefault("sensor_active_value", 1)

    # Relatórios automáticos por turno (executados pelo CORE)
    cfg.setdefault("auto_shift_reports_enabled", True)
    cfg.setdefault("auto_shift_email_enabled", True)
    cfg.setdefault("shift_report_times", ["06:00", "14:00", "22:00"])
    cfg.setdefault("shift_report_period", "today")
    cfg.setdefault("shift_report_window_min", 3)
    cfg.setdefault("shift_report_keep_state_days", 60)

    # MES / Rastreabilidade / Evidências industriais
    cfg.setdefault("mes_enabled", False)
    cfg.setdefault("traceability_enabled", False)
    cfg.setdefault("mes_xml_enabled", True)
    cfg.setdefault("block_without_serial", True)
    cfg.setdefault("clear_serial_after_inspection", True)
    cfg.setdefault("production_order", "")
    cfg.setdefault("equipment_id", "SVC01")
    cfg.setdefault("line_name", "L01")
    cfg.setdefault("product_model", "UNDEFINED")
    cfg.setdefault("serial_min_len", 4)
    cfg.setdefault("allow_duplicate_serial", False)
    cfg.setdefault("evidence_auto_enabled", True)
    cfg.setdefault("evidence_save_ok_attention", True)
    cfg.setdefault("ok_attention_confidence_max", 0.60)
    cfg.setdefault("cycle_time_logger_enabled", True)
    return cfg

def load_recipe_for_model(product_model: str):
    """Carrega a receita do produto/modelo selecionado no config_usb.json."""
    model_id = str(product_model or "").strip()

    if not model_id or model_id == "UNDEFINED":
        return None

    recipe_path = RECIPES_DIR / f"{model_id}.json"
    recipe = read_json(recipe_path, None)

    if not recipe:
        log_event("recipe_not_found", product_model=model_id, recipe_path=str(recipe_path))
        return None

    recipe["recipe_path"] = str(recipe_path)
    return recipe


def resolve_recipe_model_path(recipe: dict):
    """Resolve o caminho do modelo IA definido na receita."""
    if not recipe:
        return None

    model_path = str(recipe.get("model_path", "")).strip()

    if not model_path:
        return None

    p = Path(model_path)

    if not p.is_absolute():
        p = BASE_DIR / p

    return p

def enable_keras_legacy_batchnorm_renorm_compat():
    """Remove parametros antigos de BatchNormalization salvos por versoes antigas do Keras.

    Alguns modelos .keras antigos salvam renorm, renorm_clipping e renorm_momentum.
    Versoes atuais do Keras nao aceitam mais esses argumentos na desserializacao.
    """
    bn_classes = []

    try:
        bn_classes.append(tf.keras.layers.BatchNormalization)
    except Exception:
        pass

    try:
        import keras
        bn_classes.append(keras.layers.BatchNormalization)
    except Exception:
        pass

    for bn_cls in bn_classes:
        try:
            if getattr(bn_cls, "_svc_usb_renorm_compat_enabled", False):
                continue

            original_from_config = bn_cls.from_config

            def patched_from_config(cls, config, _original_from_config=original_from_config):
                config = dict(config)
                config.pop("renorm", None)
                config.pop("renorm_clipping", None)
                config.pop("renorm_momentum", None)
                return _original_from_config(config)

            bn_cls.from_config = classmethod(patched_from_config)
            bn_cls._svc_usb_renorm_compat_enabled = True

        except Exception:
            pass


def load_keras_model_compat(model_file):
    """Carrega modelos .keras antigos com compatibilidade para BatchNormalization."""
    enable_keras_legacy_batchnorm_renorm_compat()

    try:
        return tf.keras.models.load_model(
            str(model_file),
            compile=False,
            safe_mode=False,
        )
    except TypeError as e:
        # Algumas versoes do TensorFlow/Keras nao aceitam safe_mode.
        if "safe_mode" in str(e):
            return tf.keras.models.load_model(
                str(model_file),
                compile=False,
            )
        raise


def load_labels():
    # 3CLASS LOCK — não lê labels antigos de 4 classes.
    # O SVC USB operacional deve trabalhar apenas com:
    # OK, NG_DESALINHADO, NG_DANIFICADO.
    return list(OFFICIAL_CLASSES)


def load_model():
    labels = load_labels()
    cfg = load_config()

    product_model = str(cfg.get("product_model", "UNDEFINED")).strip()
    recipe = load_recipe_for_model(product_model)
    recipe_model_path = resolve_recipe_model_path(recipe)
    recipe_path = str(recipe.get("recipe_path", "")) if recipe else ""

    candidates = []

    if recipe_model_path:
        candidates.append(recipe_model_path)

    # Fallback legado para nao quebrar o UNICORN_WHITE enquanto migramos para receitas.
    candidates.extend([
        BASE_DIR / "outputs_usb_v15_3class" / "model_final.keras",
        BASE_DIR / "model_final.keras",
        MODELS_DIR / "model_final.keras",
        BASE_DIR / "outputs_usb_v15_3class" / "best_model.keras",
    ])

    errors = []

    for p in candidates:
        if not p.exists():
            errors.append({"path": str(p), "error": "file_not_found"})
            continue

        try:
            model = load_keras_model_compat(p)
            out_dim = int(model.output_shape[-1]) if model.output_shape[-1] is not None else None

            if out_dim == len(labels):
                model_source = "recipe" if recipe_model_path and Path(p) == Path(recipe_model_path) else "legacy_fallback"

                model_info = {
                    "product_model_loaded": product_model,
                    "recipe_path": recipe_path,
                    "recipe_model_path": str(recipe_model_path) if recipe_model_path else "",
                    "model_path": str(p),
                    "model_path_loaded": str(p),
                    "model_source": model_source,
                }

                log_event(
                    "model_loaded",
                    **model_info,
                    labels=labels,
                    output_shape=str(model.output_shape)
                )
                return model, labels, str(p), model_info

            errors.append({
                "path": str(p),
                "output_shape": str(model.output_shape),
                "labels": len(labels)
            })

        except Exception as e:
            errors.append({"path": str(p), "error": repr(e)})

    raise RuntimeError(f"Nenhum modelo compativel encontrado. Candidatos/erros: {errors}")


def open_camera(cfg):
    idx = int(cfg.get("camera_index", 0))
    backend = str(cfg.get("camera_backend", "auto")).lower()
    if backend == "dshow" and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    elif backend == "msmf" and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("camera_width", 1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("camera_height", 720)))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu câmera index={idx} backend={backend}")
    log_event("camera_opened", camera_index=idx, backend=backend)
    return cap


def safe_capture(cap, cfg):
    flush_grabs = int(cfg.get("flush_grabs", 2))
    for _ in range(max(0, flush_grabs)):
        try:
            cap.grab()
        except Exception:
            pass
    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        return frame, cap
    if bool(cfg.get("auto_reconnect_camera", True)):
        log_event("camera_read_failed_reopen")
        try:
            cap.release()
        except Exception:
            pass
        time.sleep(0.25)
        cap = open_camera(cfg)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame, cap
    raise RuntimeError("Falha ao capturar frame da câmera.")


def roi_pixels(frame, cfg):
    h, w = frame.shape[:2]
    x0 = int(max(0, min(1, float(cfg.get("roi_x0", 0.0)))) * w)
    x1 = int(max(0, min(1, float(cfg.get("roi_x1", 1.0)))) * w)
    y0 = int(max(0, min(1, float(cfg.get("roi_y0", 0.0)))) * h)
    y1 = int(max(0, min(1, float(cfg.get("roi_y1", 1.0)))) * h)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("ROI inválida.")
    return x0, y0, x1, y1


def draw_overlay(frame, cfg, result_text=None):
    img = frame.copy()
    x0, y0, x1, y1 = roi_pixels(frame, cfg)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.putText(img, "ROI USB", (x0 + 5, max(25, y0 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if result_text:
        color = (0, 190, 0) if result_text == "OK" else (0, 0, 255)
        cv2.putText(img, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4, cv2.LINE_AA)
    return img


def preprocess(bgr, img_size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA).astype("float32")
    return np.expand_dims(x, 0)


def classify(labels, probs):
    raw = {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
    pred = max(raw, key=raw.get)
    return pred, float(raw[pred]), raw


def save_images(frame, roi, overlay, result, cycle, cfg, source="camera"):
    cap_root = BASE_DIR / str(cfg.get("captures_dir", "captures_usb"))
    sub = "OK" if result["class_name"] == "OK" else "NG"
    cause = result["class_name"] if result["class_name"].startswith("NG") else "OK"
    out_dir = cap_root / sub / cause
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"USB_{ts}_{sanitize_filename(result.get('production_order','OP'))}_{sanitize_filename(result.get('serial_number','SEM_SERIAL'))}_C{cycle:06d}_{result['class_name']}_{source}"
    paths = {}
    if bool(cfg.get("save_raw_and_roi", True)):
        raw_path = out_dir / f"{stem}_raw.jpg"
        roi_path = out_dir / f"{stem}_roi.jpg"
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(roi_path), roi)
        cv2.imwrite(str(overlay_path), overlay)
        paths = {"raw": str(raw_path), "roi": str(roi_path), "overlay": str(overlay_path)}
    else:
        roi_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(roi_path), roi)
        paths = {"roi": str(roi_path)}
    cv2.imwrite(str(RUNTIME / "last_frame_overlay.jpg"), overlay)
    cv2.imwrite(str(RUNTIME / "last_roi.jpg"), roi)
    result["image_paths"] = paths
    result["overlay_path"] = str(RUNTIME / "last_frame_overlay.jpg")
    result["roi_path"] = str(RUNTIME / "last_roi.jpg")


# ==========================================================
# MES / RASTREABILIDADE / EVIDÊNCIAS
# ==========================================================
def normalize_serial_qr(serial: str) -> str:
    s = (serial or "").strip()
    s = s.replace("+", "-")
    s = re.sub(r"\s+", "", s)
    return s


def sanitize_filename(text: str) -> str:
    s = normalize_serial_qr(text)
    s = re.sub(r'[\\/*?:"<>|\s]+', "_", s)
    return (s[:120] if s else "SEM_DADO")


def current_context(cfg=None):
    cfg = cfg or load_config()
    ctx = read_json(CURRENT_CONTEXT_PATH, {}) or {}

    # LOG/SN FIX — o serial só vale para UMA inspeção quando foi marcado como pendente.
    # Isso evita que um SN antigo gravado em runtime_status/current_serial.json seja reutilizado
    # no CSV/trace/log quando a próxima peça for testada sem novo bip/scan.
    raw_sn = normalize_serial_qr(ctx.get("serial_number", ""))
    serial_pending = bool(ctx.get("serial_pending", False))
    serial_for_inspection = raw_sn if serial_pending else ""

    return {
        "timestamp": ctx.get("timestamp", now()),
        "serial_number": serial_for_inspection,
        "serial_pending": serial_pending,
        "production_order": str(ctx.get("production_order", cfg.get("production_order", ""))).strip(),
        "equipment_id": str(ctx.get("equipment_id", cfg.get("equipment_id", "SVC01"))).strip(),
        "line_name": str(ctx.get("line_name", cfg.get("line_name", "L01"))).strip(),
        "product_model": str(ctx.get("product_model", cfg.get("product_model", "UNDEFINED"))).strip(),
        "mes_enabled": bool(ctx.get("mes_enabled", cfg.get("mes_enabled", False))),
        "traceability_enabled": bool(ctx.get("traceability_enabled", cfg.get("traceability_enabled", False))),
    }


def write_current_context(ctx: dict):
    incoming = ctx or {}
    payload = current_context(load_config())
    payload.update(incoming)
    payload["timestamp"] = now()
    payload["serial_number"] = normalize_serial_qr(payload.get("serial_number", ""))

    # Quando o app envia um novo serial_number, ele passa a ficar pendente para a próxima inspeção.
    # Quando envia serial vazio, o core limpa o pendente. Campos de OP/linha/equipamento continuam preservados.
    if "serial_number" in incoming:
        payload["serial_pending"] = bool(payload["serial_number"])

    atomic_write_json(CURRENT_CONTEXT_PATH, payload)
    return payload


def clear_current_serial():
    ctx = current_context(load_config())
    ctx["serial_number"] = ""
    ctx["serial_pending"] = False
    ctx["timestamp"] = now()
    atomic_write_json(CURRENT_CONTEXT_PATH, ctx)
    return ctx


def validate_operation_context(cfg, ctx):
    trace_on = bool(cfg.get("traceability_enabled", False)) or bool(cfg.get("mes_enabled", False)) or bool(ctx.get("traceability_enabled", False)) or bool(ctx.get("mes_enabled", False))
    if bool(cfg.get("mes_enabled", False)) and not (bool(cfg.get("traceability_enabled", False)) or bool(ctx.get("traceability_enabled", False))):
        raise RuntimeError("MES ativo exige rastreabilidade ativa.")
    if trace_on and bool(cfg.get("block_without_serial", True)):
        sn = normalize_serial_qr(ctx.get("serial_number", ""))
        min_len = int(cfg.get("serial_min_len", 4))
        if len(sn) < min_len:
            raise RuntimeError("Rastreabilidade ativa: escaneie o número de série/QRCode antes da inspeção.")
        if not str(ctx.get("production_order", "")).strip():
            raise RuntimeError("Rastreabilidade ativa: informe a Ordem de Produção antes da inspeção.")
        if not str(ctx.get("equipment_id", "")).strip():
            raise RuntimeError("Rastreabilidade ativa: informe o Equipment ID antes da inspeção.")
    return True


def generate_inspection_id():
    return datetime.now().strftime("INSP_%Y%m%d_%H%M%S_%f")


def create_mes_xml(result, ctx, cfg):
    inspection_id = result.get("inspection_id") or generate_inspection_id()
    root = ET.Element("inspection")
    fields = {
        "timestamp": result.get("timestamp", now()),
        "inspection_id": inspection_id,
        "system_name": "SVC USB — Computer Vision System for USB Inspection",
        "equipment_id": ctx.get("equipment_id", ""),
        "line": ctx.get("line_name", ""),
        "production_order": ctx.get("production_order", ""),
        "serial_number": ctx.get("serial_number", ""),
        "model_name": ctx.get("product_model", ""),
        "mes_enabled": str(bool(ctx.get("mes_enabled", False))).lower(),
        "traceability_enabled": str(bool(ctx.get("traceability_enabled", False))).lower(),
        "source": result.get("source", ""),
        "cycle": str(result.get("cycle", "")),
        "decision": result.get("decision", ""),
        "final_result": "PASS" if result.get("ok") else "FAIL",
        "class_name": result.get("class_name", ""),
        "confidence": f"{float(result.get('confidence', 0.0)):.6f}",
        "image_roi": result.get("image_paths", {}).get("roi", result.get("roi_path", "")),
        "image_overlay": result.get("image_paths", {}).get("overlay", result.get("overlay_path", "")),
        "mes_status": "LOCAL_XML_PENDING",
    }
    for k, v in fields.items():
        child = ET.SubElement(root, k)
        child.text = str(v)
    xml_name = f"{inspection_id}_{sanitize_filename(ctx.get('serial_number',''))}.xml"
    xml_path = MES_XML_DIR / xml_name
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return str(xml_path)


def append_trace_log(result, ctx):
    fields = [
        "timestamp", "inspection_id", "cycle", "source", "serial_number", "production_order", "equipment_id", "line_name", "product_model",
        "mes_enabled", "traceability_enabled", "decision", "class_name", "confidence", "ok", "image_roi", "image_overlay", "xml_path", "mes_status"
    ]
    exists = TRACE_LOG_PATH.exists()
    with open(TRACE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp": result.get("timestamp", ""),
            "inspection_id": result.get("inspection_id", ""),
            "cycle": result.get("cycle", ""),
            "source": result.get("source", ""),
            "serial_number": result.get("serial_number", ""),
            "production_order": result.get("production_order", ctx.get("production_order", "")),
            "equipment_id": ctx.get("equipment_id", ""),
            "line_name": ctx.get("line_name", ""),
            "product_model": ctx.get("product_model", ""),
            "mes_enabled": ctx.get("mes_enabled", False),
            "traceability_enabled": ctx.get("traceability_enabled", False),
            "decision": result.get("decision", ""),
            "class_name": result.get("class_name", ""),
            "confidence": f"{float(result.get('confidence',0.0)):.6f}",
            "ok": result.get("ok", ""),
            "image_roi": result.get("image_paths", {}).get("roi", result.get("roi_path", "")),
            "image_overlay": result.get("image_paths", {}).get("overlay", result.get("overlay_path", "")),
            "xml_path": result.get("xml_path", ""),
            "mes_status": result.get("mes_status", "LOCAL"),
        })


def ensure_csv_header(fields):
    if not CSV_LOG_PATH.exists():
        return False
    try:
        with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
            first = f.readline().strip().split(";")
        missing = [x for x in fields if x not in first]
        if missing:
            backup = CSV_LOG_PATH.with_name(f"inspection_log_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            shutil.copy2(CSV_LOG_PATH, backup)
            log_event("csv_header_backup_created", backup=str(backup), missing=missing)
            CSV_LOG_PATH.unlink()
            return False
    except Exception as e:
        log_event("csv_header_check_error", error=repr(e))
    return CSV_LOG_PATH.exists()


def update_summary(result):
    summary = read_json(SUMMARY_PATH, {"timestamp": now(), "total": 0, "ok": 0, "ng": 0, "classes": {}}) or {"total": 0, "ok": 0, "ng": 0, "classes": {}}
    summary["timestamp"] = now()
    summary["total"] = int(summary.get("total", 0)) + 1
    if result["class_name"] == "OK":
        summary["ok"] = int(summary.get("ok", 0)) + 1
    else:
        summary["ng"] = int(summary.get("ng", 0)) + 1
    classes = summary.setdefault("classes", {})
    classes[result["class_name"]] = int(classes.get(result["class_name"], 0)) + 1
    total = max(1, int(summary["total"]))
    summary["yield_pct"] = round(100 * int(summary.get("ok", 0)) / total, 2)
    atomic_write_json(SUMMARY_PATH, summary)
    return summary


def append_csv(result):
    fields = [
        "timestamp", "cycle", "source", "serial_number", "production_order", "equipment_id", "line_name", "product_model", "inspection_id",
        "decision", "class_name", "confidence", "ok", "probs", "roi", "latency_total_ms", "cycle_qr_to_result_ms", "cycle_capture_ms", "cycle_inference_pipeline_ms", "cycle_postprocess_ms", "image_roi", "image_overlay", "xml_path", "mes_status"
    ]
    exists = ensure_csv_header(fields)
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp": result["timestamp"],
            "cycle": result["cycle"],
            "source": result["source"],
            "serial_number": result.get("serial_number", ""),
            "production_order": result.get("production_order", ""),
            "equipment_id": result.get("equipment_id", ""),
            "line_name": result.get("line_name", ""),
            "product_model": result.get("product_model", ""),
            "inspection_id": result.get("inspection_id", ""),
            "decision": result["decision"],
            "class_name": result["class_name"],
            "confidence": f'{result["confidence"]:.6f}',
            "ok": result["ok"],
            "probs": json.dumps(result.get("probs", {}), ensure_ascii=False),
            "roi": json.dumps(result.get("roi", [])),
            "latency_total_ms": result.get("latency", {}).get("total"),
            "cycle_qr_to_result_ms": result.get("cycle_time", {}).get("qr_to_result_ms", ""),
            "cycle_capture_ms": result.get("cycle_time", {}).get("capture_ms", ""),
            "cycle_inference_pipeline_ms": result.get("cycle_time", {}).get("inference_pipeline_ms", ""),
            "cycle_postprocess_ms": result.get("cycle_time", {}).get("postprocess_to_result_ms", ""),
            "image_roi": result.get("image_paths", {}).get("roi", ""),
            "image_overlay": result.get("image_paths", {}).get("overlay", ""),
            "xml_path": result.get("xml_path", ""),
            "mes_status": result.get("mes_status", ""),
        })


def inspect_frame(frame, model, labels, cfg, source, cycle, cycle_timing=None):
    ctx = current_context(cfg)
    validate_operation_context(cfg, ctx)
    inspection_id = generate_inspection_id()
    cycle_timing = dict(cycle_timing or {})
    cycle_timing.setdefault("inspect_frame_start_epoch_ms", epoch_ms())
    t0 = time.perf_counter()
    x0, y0, x1, y1 = roi_pixels(frame, cfg)
    roi = frame[y0:y1, x0:x1].copy()
    t1 = time.perf_counter()
    x = preprocess(roi, int(cfg.get("img_size", 224)))
    probs = np.asarray(model.predict(x, verbose=0)[0], dtype=float)
    pred, conf, raw = classify(labels, probs)
    decision = "OK" if pred == "OK" else "REPROVADO"
    t2 = time.perf_counter()
    infer_done_epoch_ms = epoch_ms()
    cycle_timing.setdefault("infer_done_epoch_ms", infer_done_epoch_ms)
    cycle_time = {
        "qr_scan_at": cycle_timing.get("qr_scan_at", ""),
        "qr_scan_epoch_ms": cycle_timing.get("qr_scan_epoch_ms"),
        "command_epoch_ms": cycle_timing.get("command_epoch_ms"),
        "command_seen_epoch_ms": cycle_timing.get("command_seen_epoch_ms"),
        "capture_begin_epoch_ms": cycle_timing.get("capture_begin_epoch_ms"),
        "capture_done_epoch_ms": cycle_timing.get("capture_done_epoch_ms"),
        "inspect_frame_start_epoch_ms": cycle_timing.get("inspect_frame_start_epoch_ms"),
        "infer_done_epoch_ms": cycle_timing.get("infer_done_epoch_ms"),
        "qr_to_command_seen_ms": None,
        "command_to_capture_begin_ms": None,
        "capture_ms": None,
        "capture_to_infer_done_ms": None,
        "inference_pipeline_ms": round((t2 - t0) * 1000, 2),
        "roi_crop_ms": round((t1 - t0) * 1000, 2),
        "model_inference_ms": round((t2 - t1) * 1000, 2),
    }
    try:
        if cycle_time["qr_scan_epoch_ms"] is not None and cycle_time["command_seen_epoch_ms"] is not None:
            cycle_time["qr_to_command_seen_ms"] = round(float(cycle_time["command_seen_epoch_ms"]) - float(cycle_time["qr_scan_epoch_ms"]), 2)
        if cycle_time["command_epoch_ms"] is not None and cycle_time["capture_begin_epoch_ms"] is not None:
            cycle_time["command_to_capture_begin_ms"] = round(float(cycle_time["capture_begin_epoch_ms"]) - float(cycle_time["command_epoch_ms"]), 2)
        if cycle_time["capture_begin_epoch_ms"] is not None and cycle_time["capture_done_epoch_ms"] is not None:
            cycle_time["capture_ms"] = round(float(cycle_time["capture_done_epoch_ms"]) - float(cycle_time["capture_begin_epoch_ms"]), 2)
        if cycle_time["capture_done_epoch_ms"] is not None:
            cycle_time["capture_to_infer_done_ms"] = round(float(cycle_time["infer_done_epoch_ms"]) - float(cycle_time["capture_done_epoch_ms"]), 2)
    except Exception as e:
        cycle_time["cycle_time_calc_error"] = repr(e)
    result = {
        "timestamp": now(),
        "cycle": cycle,
        "source": source,
        "inspection_id": inspection_id,
        "serial_number": ctx.get("serial_number", ""),
        "production_order": ctx.get("production_order", ""),
        "equipment_id": ctx.get("equipment_id", ""),
        "line_name": ctx.get("line_name", ""),
        "product_model": ctx.get("product_model", ""),
        "mes_enabled": ctx.get("mes_enabled", False),
        "traceability_enabled": ctx.get("traceability_enabled", False),
        "decision": decision,
        "class_name": pred,
        "confidence": conf,
        "ok": pred == "OK",
        "probs": raw,
        "roi": [x0, y0, x1, y1],
        "core_version": CORE_VERSION,
        "latency": {"capture": round((t1 - t0) * 1000, 2), "inference": round((t2 - t1) * 1000, 2), "total": round((t2 - t0) * 1000, 2)},
        "cycle_time": cycle_time,
    }
    overlay = draw_overlay(frame, cfg, decision)
    should_save = bool(cfg.get("save_all_captures", False)) or pred != "OK" or bool(cfg.get("save_ng_images", True))
    if should_save:
        save_images(frame, roi, overlay, result, cycle, cfg, source)
    else:
        cv2.imwrite(str(RUNTIME / "last_frame_overlay.jpg"), overlay)
        cv2.imwrite(str(RUNTIME / "last_roi.jpg"), roi)
        result["overlay_path"] = str(RUNTIME / "last_frame_overlay.jpg")
        result["roi_path"] = str(RUNTIME / "last_roi.jpg")
    result["mes_status"] = "LOCAL"
    result["xml_path"] = ""
    if bool(cfg.get("mes_xml_enabled", True)) and (bool(ctx.get("mes_enabled")) or bool(ctx.get("traceability_enabled"))):
        try:
            result["xml_path"] = create_mes_xml(result, ctx, cfg)
            result["mes_status"] = "LOCAL_XML_PENDING"
        except Exception as e:
            log_event("mes_xml_error", cycle=cycle, error=repr(e))
            result["mes_status"] = "XML_ERROR"
    result_epoch_ms = epoch_ms()
    result["cycle_time"]["result_available_epoch_ms"] = result_epoch_ms
    try:
        if result["cycle_time"].get("qr_scan_epoch_ms") is not None:
            result["cycle_time"]["qr_to_result_ms"] = round(result_epoch_ms - float(result["cycle_time"].get("qr_scan_epoch_ms")), 2)
        if result["cycle_time"].get("command_epoch_ms") is not None:
            result["cycle_time"]["command_to_result_ms"] = round(result_epoch_ms - float(result["cycle_time"].get("command_epoch_ms")), 2)
        result["cycle_time"]["postprocess_to_result_ms"] = round(result_epoch_ms - float(result["cycle_time"].get("infer_done_epoch_ms", result_epoch_ms)), 2)
    except Exception as e:
        result["cycle_time"]["cycle_time_total_calc_error"] = repr(e)
    atomic_write_json(LAST_RESULT_PATH, result)
    update_summary(result)
    append_csv(result)
    append_trace_log(result, ctx)
    log_event("infer_end", source=source, cycle=cycle, decision=decision, result=pred, confidence=conf, total_ms=result["latency"]["total"], cycle_time=result.get("cycle_time", {}), roi=result["roi"], serial_number=result.get("serial_number", ""), production_order=result.get("production_order", ""), xml_path=result.get("xml_path", ""))

    # LOG/SN FIX — consome o SN após gravar last_result/CSV/trace/XML.
    # Na próxima inspeção, se não houver novo bip/scan, o log será gravado com serial_number em branco.
    if bool(cfg.get("clear_serial_after_inspection", True)):
        sn_used = normalize_serial_qr(result.get("serial_number", ""))
        clear_current_serial()
        if sn_used:
            log_event("serial_consumed_after_inspection", cycle=cycle, inspection_id=inspection_id, serial_number=sn_used)
        else:
            log_event("serial_blank_logged_after_inspection", cycle=cycle, inspection_id=inspection_id)
    return result


def write_heartbeat(state, cycle=0, extra=None):
    """Atualiza o heartbeat sem derrubar o core.

    Em Windows, o arquivo heartbeat.json pode ficar bloqueado por leitura
    momentanea do Streamlit, antivirus ou Explorer. Esse arquivo e apenas
    diagnostico; falha de escrita nele nao pode parar a inspecao.
    """
    try:
        config_mtime = CONFIG_PATH.stat().st_mtime
    except Exception:
        config_mtime = None

    payload = {
        "timestamp": now(),
        "state": state,
        "cycle": cycle,
        "core_version": CORE_VERSION,
        "pid": os.getpid(),
        "core_started_at_epoch": CORE_STARTED_AT_EPOCH,
        "config_mtime": config_mtime,
    }

    if extra:
        payload.update(extra)

    try:
        atomic_write_json(HEARTBEAT_PATH, payload, retries=5, delay=0.02)
    except PermissionError as e:
        try:
            log_event("heartbeat_write_skipped", error=repr(e), state=state, cycle=cycle)
        except Exception:
            pass
    except Exception as e:
        try:
            log_event("heartbeat_write_error", error=repr(e), state=state, cycle=cycle)
        except Exception:
            pass


def handle_dataset_copy(req):
    cls = str(req.get("class_name", "OK")).upper().strip()
    cfg = load_config()
    allowed = set(cfg.get("dataset_classes", OFFICIAL_CLASSES))
    if cls not in allowed:
        raise RuntimeError(f"Classe inválida: {cls}")
    last = read_json(LAST_RESULT_PATH, {}) or {}
    src_roi = req.get("image_path") or last.get("image_paths", {}).get("roi") or last.get("roi_path")
    if not src_roi or not Path(src_roi).exists():
        raise RuntimeError("Nenhuma ROI disponível para salvar.")
    ds_root = BASE_DIR / str(cfg.get("dataset_dir", "dataset_usb_live_capture"))
    out_dir = ds_root / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst = out_dir / f"USB_{ts}_{cls}.jpg"
    shutil.copy2(src_roi, dst)
    atomic_write_json(dst.with_suffix(".json"), {"timestamp": now(), "class_name": cls, "source_image": str(src_roi), "dest_image": str(dst), "last_result": last})
    log_event("dataset_saved", class_name=cls, dest=str(dst))
    return {"saved_path": str(dst), "class_name": cls}


def read_csv_rows(period="today"):
    if not CSV_LOG_PATH.exists():
        return []
    rows = []
    with open(CSV_LOG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if period == "today":
                try:
                    dt = datetime.strptime(row.get("timestamp", "")[:10], "%Y-%m-%d").date()
                    if dt != date.today():
                        continue
                except Exception:
                    pass
            rows.append(row)
    return rows


def handle_report(req):
    period = req.get("period", "today")
    rows = read_csv_rows(period)
    total = len(rows)
    ok = sum(1 for r in rows if r.get("class_name") == "OK")
    ng = total - ok

    classes = {}
    for r in rows:
        cls = r.get("class_name", "-")
        classes[cls] = classes.get(cls, 0) + 1

    yield_pct = round(100 * ok / max(total, 1), 2)
    ng_rows = [r for r in rows if str(r.get("class_name", "")).startswith("NG")]
    ng_total = max(len(ng_rows), 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"relatorio_svc_usb_{ts}.html"
    pdf_path = REPORTS_DIR / f"relatorio_svc_usb_{ts}.pdf"

    # ==========================================================
    # HTML = RESUMO EXECUTIVO PARA CORPO/E-MAIL
    # ==========================================================
    cfg = load_config()
    ctx = current_context(cfg)
    shift_time = str(req.get("shift_time", "")).strip()
    generated_at = now()
    report_context = "Relatório automático de turno" if str(req.get("source", "")).lower() == "auto_shift_report" else "Relatório de Auditoria"
    subtitle = f"{report_context} - {shift_time}" if shift_time else report_context

    # Garante que as classes oficiais apareçam no resumo mesmo quando a contagem for zero.
    display_classes = []
    for cls_name in OFFICIAL_CLASSES:
        if cls_name not in display_classes:
            display_classes.append(cls_name)
    for cls_name in sorted(classes.keys()):
        if cls_name not in display_classes:
            display_classes.append(cls_name)

    def _pct_total(value):
        return round(100 * int(value) / max(total, 1), 2)

    def _pct_ng(value, cls_name):
        return round(100 * int(value) / ng_total, 2) if str(cls_name).startswith("NG") else "-"

    class_rows_html = "".join(
        f"<tr>"
        f"<td style='padding:8px;border:1px solid #d0d7de'>{cls_name}</td>"
        f"<td style='padding:8px;border:1px solid #d0d7de;text-align:center'>{int(classes.get(cls_name, 0))}</td>"
        f"<td style='padding:8px;border:1px solid #d0d7de;text-align:center'>{_pct_total(classes.get(cls_name, 0))}%</td>"
        f"<td style='padding:8px;border:1px solid #d0d7de;text-align:center'>{_pct_ng(classes.get(cls_name, 0), cls_name)}</td>"
        f"</tr>"
        for cls_name in display_classes
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>Resumo do Relatório de Auditoria - SVC USB</title>
</head>
<body style="font-family:Arial,Helvetica,sans-serif;color:#111827;margin:0;padding:24px;background:#ffffff">
  <div style="max-width:760px;margin:0 auto">
    <h2 style="margin:0 0 16px 0;color:#0b3b75;font-size:22px">
      Resumo do Relatório de Auditoria – SVC USB
    </h2>

    <p style="margin:0 0 14px 0;font-size:14px;line-height:1.45">
      <b>Tipo:</b> {subtitle}<br>
      <b>Emitido em:</b> {generated_at}<br>
      <b>Período:</b> {period}<br>
      <b>Linha:</b> {ctx.get('line_name') or '---'} &nbsp; | &nbsp;
      <b>Equipamento:</b> {ctx.get('equipment_id') or '---'}<br>
      <b>Modelo:</b> {ctx.get('product_model') or '---'} &nbsp; | &nbsp;
      <b>OP:</b> {ctx.get('production_order') or '---'}
    </p>

    <table style="border-collapse:collapse;margin:12px 0 18px 0;font-size:14px">
      <tr>
        <th style="padding:8px 12px;border:1px solid #9ca3af;background:#f3f4f6">Total</th>
        <th style="padding:8px 12px;border:1px solid #9ca3af;background:#f3f4f6">OK</th>
        <th style="padding:8px 12px;border:1px solid #9ca3af;background:#f3f4f6">NG</th>
        <th style="padding:8px 12px;border:1px solid #9ca3af;background:#f3f4f6">Yield</th>
      </tr>
      <tr>
        <td style="padding:10px 16px;border:1px solid #9ca3af;text-align:center">{total}</td>
        <td style="padding:10px 16px;border:1px solid #9ca3af;text-align:center;color:#087c22;font-weight:bold">{ok}</td>
        <td style="padding:10px 16px;border:1px solid #9ca3af;text-align:center;color:#b00000;font-weight:bold">{ng}</td>
        <td style="padding:10px 16px;border:1px solid #9ca3af;text-align:center;font-weight:bold">{yield_pct}%</td>
      </tr>
    </table>

    <p style="margin:10px 0 6px 0;font-size:15px"><b>Falhas / causas prováveis:</b></p>
    <table style="border-collapse:collapse;width:100%;max-width:620px;font-size:13px">
      <tr>
        <th style="padding:8px;border:1px solid #d0d7de;background:#f3f4f6;text-align:left">Classe</th>
        <th style="padding:8px;border:1px solid #d0d7de;background:#f3f4f6">Quantidade</th>
        <th style="padding:8px;border:1px solid #d0d7de;background:#f3f4f6">% Total</th>
        <th style="padding:8px;border:1px solid #d0d7de;background:#f3f4f6">% entre NG</th>
      </tr>
      {class_rows_html}
    </table>

    <p style="margin:18px 0 0 0;font-size:13px;line-height:1.45;color:#374151">
      Relatório completo em HTML e PDF anexos.<br>
      Relatório gerado automaticamente pelo sistema <b>SVC USB v2.1 Production</b>.
    </p>
  </div>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")

    # ==========================================================
    # PDF = RELATÓRIO DETALHADO DE AUDITORIA
    # ==========================================================
    pdf_created = False
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm

        styles = getSampleStyleSheet()
        small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=7, leading=9)
        normal = styles["Normal"]
        title = styles["Title"]
        h2 = styles["Heading2"]

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=landscape(A4),
            rightMargin=1.0 * cm,
            leftMargin=1.0 * cm,
            topMargin=1.0 * cm,
            bottomMargin=1.0 * cm,
        )

        story = []
        story.append(Paragraph("Relatório Detalhado de Auditoria — SVC USB v2.1", title))
        story.append(Paragraph(f"Gerado em: {now()} | Período: {period}", normal))
        story.append(Paragraph("Sistema: SVC USB — Computer Vision System for USB Inspection", normal))
        story.append(Spacer(1, 10))

        resumo_data = [
            ["Total", "OK", "NG", "Yield"],
            [str(total), str(ok), str(ng), f"{yield_pct}%"],
        ]
        resumo_table = Table(resumo_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm])
        resumo_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story.append(resumo_table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Causas prováveis", h2))
        cause_data = [["Classe", "Quantidade", "% Total", "% entre NG"]]
        for k, v in sorted(classes.items()):
            cause_data.append([
                str(k),
                str(v),
                f"{round(100*v/max(total,1),2)}%",
                f"{round(100*v/ng_total,2)}%" if str(k).startswith("NG") else "-",
            ])
        cause_table = Table(cause_data, colWidths=[7*cm, 3*cm, 3*cm, 3*cm])
        cause_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story.append(cause_table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Últimas inspeções detalhadas", h2))
        story.append(Paragraph("Observação: a decisão operacional é OK/REPROVADO; a classe NG representa causa provável para análise de processo.", small))
        story.append(Spacer(1, 6))

        detail_data = [["Data/hora", "Ciclo", "Serial", "OP", "Equip.", "Decisão", "Causa provável", "Conf.", "Imagem ROI"]]
        for r in rows[-120:]:
            img_roi = str(r.get("image_roi", ""))
            # quebra caminho longo para não estourar a página
            img_roi = img_roi.replace("\\", "\\ ")
            detail_data.append([
                Paragraph(str(r.get("timestamp", "")), small),
                Paragraph(str(r.get("cycle", "")), small),
                Paragraph(str(r.get("serial_number", "")), small),
                Paragraph(str(r.get("production_order", "")), small),
                Paragraph(str(r.get("equipment_id", "")), small),
                Paragraph(str(r.get("decision", "")), small),
                Paragraph(str(r.get("class_name", "")), small),
                Paragraph(str(r.get("confidence", "")), small),
                Paragraph(img_roi, small),
            ])

        detail_table = Table(detail_data, repeatRows=1, colWidths=[3.6*cm, 1.1*cm, 4.0*cm, 3.0*cm, 1.8*cm, 2.2*cm, 3.2*cm, 1.5*cm, 8.0*cm])
        detail_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(detail_table)

        doc.build(story)
        pdf_created = True
    except Exception as e:
        log_event("pdf_generation_skipped", error=repr(e))

    log_event("report_generated", html=str(html_path), pdf=str(pdf_path) if pdf_created else "")
    return {"report_html": str(html_path), "report_pdf": str(pdf_path) if pdf_created else ""}

def _read_explicit_email_html(req):
    """Lê HTML somente quando o chamador pedir explicitamente.

    Mantém o envio manual existente intacto: anexar um .html não muda sozinho
    o corpo do e-mail. O fluxo automático por turno passa html_path explicitamente.
    """
    html_body = req.get("html_body") or req.get("body_html")
    if html_body:
        return str(html_body)

    html_path = req.get("html_path") or req.get("report_html")
    if html_path and Path(str(html_path)).exists():
        try:
            return Path(str(html_path)).read_text(encoding="utf-8")
        except Exception as e:
            log_event("email_html_read_error", path=str(html_path), error=repr(e))
    return ""


def send_email(req):
    cfg = read_json(EMAIL_CONFIG_PATH, {}) or {}
    to = req.get("to") or cfg.get("to") or cfg.get("email_to")
    if isinstance(to, list):
        to = ", ".join(to)
    if not to:
        raise RuntimeError("Nenhum destinatário configurado.")

    msg = EmailMessage()
    msg["Subject"] = req.get("subject") or cfg.get("subject", "Relatório SVC USB")
    msg["From"] = cfg.get("smtp_user", "")
    msg["To"] = to

    plain_body = req.get("body", "Segue relatório do SVC USB.")
    html_body = _read_explicit_email_html(req)

    # Mantém texto simples como fallback e adiciona HTML somente quando solicitado.
    msg.set_content(str(plain_body))
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    # Aceita 1 anexo ("attachment") ou múltiplos anexos ("attachments")
    attachments = []
    if req.get("attachments"):
        attachments = req.get("attachments") if isinstance(req.get("attachments"), list) else [req.get("attachments")]
    elif req.get("attachment"):
        attachments = [req.get("attachment")]

    attached_files = []
    for attach in attachments:
        if attach and Path(str(attach)).exists():
            data = Path(str(attach)).read_bytes()
            ctype = mimetypes.guess_type(str(attach))[0] or "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=Path(str(attach)).name)
            attached_files.append(str(attach))

    server = cfg.get("smtp_server", "smtp.office365.com")
    port = int(cfg.get("smtp_port", 587))
    user = cfg.get("smtp_user", "")
    pwd = cfg.get("smtp_password", "")
    with smtplib.SMTP(server, port, timeout=30) as s:
        if cfg.get("smtp_use_tls", True):
            s.starttls(context=ssl.create_default_context())
        if user:
            s.login(user, pwd)
        s.send_message(msg)
    log_event("email_sent", to=to, attachments=attached_files, html_body=bool(html_body))
    return {"email_sent_to": to, "attachments": attached_files, "html_body": bool(html_body)}


def cleanup_disk(req):
    cfg = load_config()
    keep_days = int(req.get("keep_days") or cfg.get("retention_days", 30))
    cutoff = time.time() - keep_days * 86400
    roots = [BASE_DIR / str(cfg.get("captures_dir", "captures_usb")), LOG_DIR, REPORTS_DIR]
    removed = 0
    bytes_removed = 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.stat().st_mtime < cutoff:
                try:
                    size = p.stat().st_size
                    p.unlink()
                    removed += 1
                    bytes_removed += size
                except Exception as e:
                    log_event("cleanup_error", file=str(p), error=repr(e))
    log_event("cleanup_done", keep_days=keep_days, removed=removed, bytes_removed=bytes_removed)
    return {"removed": removed, "bytes_removed": bytes_removed, "keep_days": keep_days}




def parse_sensor_present(line: str):
    """Aceita linhas simples do Arduino: 0/1, PRESENT=0/1, SENSOR:0/1."""
    if line is None:
        return None
    s = str(line).strip().upper()
    if not s:
        return None
    # prefer last explicit 0/1 in line
    digits = [ch for ch in s if ch in "01"]
    if not digits:
        return None
    return int(digits[-1])


def open_serial_if_needed(cfg, ser):
    if not bool(cfg.get("serial_enabled", False)):
        if ser is not None:
            try:
                ser.close()
                log_event("serial_closed")
            except Exception:
                pass
        return None, {"serial_state": "OFF", "serial_error": ""}
    if serial is None:
        return None, {"serial_state": "ERROR", "serial_error": "pyserial não instalado"}
    port = str(cfg.get("serial_port", "COM1"))
    baud = int(cfg.get("serial_baud", 115200))
    if ser is not None and getattr(ser, "is_open", False):
        # if port changed, reopen
        try:
            if str(ser.port).upper() == port.upper() and int(ser.baudrate) == baud:
                return ser, {"serial_state": "ON", "serial_error": ""}
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=1)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        log_event("serial_opened", port=port, baud=baud)
        return ser, {"serial_state": "ON", "serial_error": ""}
    except Exception as e:
        log_event("serial_open_error", port=port, baud=baud, error=repr(e))
        return None, {"serial_state": "ERROR", "serial_error": repr(e)}


def read_sensor_line(ser):
    """
    Leitura serial robusta para Arduino.
    IMPORTANTE: não usa in_waiting, pois em alguns drivers Windows/Arduino
    o buffer pode não atualizar de forma confiável. Usa readline com timeout.
    """
    if ser is None:
        return None
    try:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        return raw or None
    except Exception as e:
        log_event("serial_read_error", error=repr(e))
        return None


def maybe_fire_from_sensor(sensor_state, present, cfg):
    """Retorna True quando deve disparar inspeção por sensor."""
    now_ts = time.time()
    active = int(cfg.get("sensor_active_value", 1))
    stable_ms = int(cfg.get("serial_stable_ms", 80))
    rearm_ms = int(cfg.get("serial_rearm_ms", 600))
    mode = str(cfg.get("trigger_mode", "edge_0to1"))

    if present is None:
        return False

    if sensor_state.get("raw_present") != present:
        sensor_state["raw_present"] = present
        sensor_state["changed_at"] = now_ts

    stable_for_ms = (now_ts - float(sensor_state.get("changed_at", now_ts))) * 1000.0
    if stable_for_ms < stable_ms:
        return False

    last_stable = sensor_state.get("stable_present")
    if last_stable != present:
        sensor_state["prev_stable_present"] = last_stable
        sensor_state["stable_present"] = present

    last_fire = float(sensor_state.get("last_fire_ts", 0.0))
    if (now_ts - last_fire) * 1000.0 < rearm_ms:
        return False

    if mode == "stable_high":
        if present == active:
            sensor_state["last_fire_ts"] = now_ts
            sensor_state["last_sensor_fire"] = f"stable_high present={present}"
            return True
    else:  # edge_0to1 default
        prev = sensor_state.get("prev_stable_present")
        if prev is not None and int(prev) != active and int(present) == active:
            sensor_state["last_fire_ts"] = now_ts
            sensor_state["last_sensor_fire"] = f"edge_0to1 {prev}->{present}"
            # consume edge
            sensor_state["prev_stable_present"] = present
            return True
    return False

def _normalize_shift_times(times):
    """Normaliza horários HH:MM vindos do JSON."""
    out = []
    if not isinstance(times, list):
        return out
    for t in times:
        s = str(t).strip()
        try:
            hh, mm = s.split(":")[:2]
            out.append(f"{int(hh):02d}:{int(mm):02d}")
        except Exception:
            continue
    return sorted(set(out))


def _cleanup_auto_report_state(state, keep_days=60):
    """Remove chaves antigas do runtime_status/auto_report_state.json."""
    try:
        cutoff = (date.today() - timedelta(days=int(keep_days))).strftime("%Y-%m-%d")
        keep = {}
        for k, v in (state or {}).items():
            day = str(k).split("_")[0]
            if day >= cutoff:
                keep[k] = v
        return keep
    except Exception:
        return state or {}


def maybe_auto_shift_report(cfg):
    """
    Gera e envia relatório automaticamente no final de cada turno.
    O agendamento fica no config_usb.json; o estado fica em runtime_status/auto_report_state.json
    para evitar envio duplicado após refresh/restart.
    """
    if not bool(cfg.get("auto_shift_reports_enabled", False)):
        return

    times = _normalize_shift_times(cfg.get("shift_report_times", []))
    if not times:
        return

    now_dt = datetime.now()
    current_hm = now_dt.strftime("%H:%M")
    if current_hm not in times:
        return

    state = read_json(AUTO_REPORT_STATE_PATH, {}) or {}
    state = _cleanup_auto_report_state(state, int(cfg.get("shift_report_keep_state_days", 60)))

    key = f"{now_dt.strftime('%Y-%m-%d')}_{current_hm}"
    if state.get(key, {}).get("status") == "sent":
        return

    # Janela de segurança: só dispara nos primeiros minutos após HH:MM
    # Ex.: se o core iniciou 14:08, não envia retroativo de 14:00.
    window_min = int(cfg.get("shift_report_window_min", 3))
    if int(now_dt.strftime("%M")) != int(current_hm.split(":")[1]):
        return
    if now_dt.second > max(0, min(59, window_min * 60 - 1)):
        return

    log_event("auto_shift_report_due", key=key, time=current_hm)
    try:
        period = str(cfg.get("shift_report_period", "today"))
        report = handle_report({"period": period, "source": "auto_shift_report", "shift_time": current_hm})
        attachments = [p for p in [report.get("report_html"), report.get("report_pdf")] if p and Path(p).exists()]

        email_sent = False
        if bool(cfg.get("auto_shift_email_enabled", True)):
            subject = f"[SVC USB] Relatório automático de turno - {current_hm}"
            body = (
                f"Segue relatório automático do SVC USB gerado ao final do turno de {current_hm}.\n\n"
                f"Período: {period}\n"
                f"Arquivos anexos: HTML resumo e PDF detalhado de auditoria."
            )
            send_email({
                "subject": subject,
                "body": body,
                "html_path": report.get("report_html", ""),
                "attachments": attachments,
            })
            email_sent = True

        state[key] = {
            "status": "sent",
            "timestamp": now(),
            "shift_time": current_hm,
            "report_html": report.get("report_html", ""),
            "report_pdf": report.get("report_pdf", ""),
            "email_sent": email_sent,
        }
        atomic_write_json(AUTO_REPORT_STATE_PATH, state)
        log_event("auto_shift_report_sent", key=key, shift_time=current_hm, attachments=attachments, email_sent=email_sent)
    except Exception as e:
        state[key] = {
            "status": "error",
            "timestamp": now(),
            "shift_time": current_hm,
            "error": repr(e),
        }
        atomic_write_json(AUTO_REPORT_STATE_PATH, state)
        log_event("auto_shift_report_error", key=key, error=repr(e), traceback=traceback.format_exc())


def execute_command(cmd, cap, model, labels, cycle):
    name = cmd.get("command")
    cfg = load_config()
    if name == "set_context":
        ctx = write_current_context(cmd.get("context", cmd))
        return cap, {"ok": True, "context_saved": ctx}
    if name == "clear_serial":
        ctx = clear_current_serial()
        return cap, {"ok": True, "serial_cleared": True, "context": ctx}
    if name in ("inspect_once", "test_once"):
        command_seen_epoch_ms = epoch_ms()
        cycle_timing = {
            "qr_scan_epoch_ms": cmd.get("qr_scan_epoch_ms"),
            "qr_scan_at": cmd.get("qr_scan_at", ""),
            "command_epoch_ms": cmd.get("command_epoch_ms"),
            "command_seen_epoch_ms": command_seen_epoch_ms,
        }
        log_event("capture_begin", source=cmd.get("source", "manual"), cycle=cycle, cycle_timing=cycle_timing)
        cycle_timing["capture_begin_epoch_ms"] = epoch_ms()
        frame, cap = safe_capture(cap, cfg)
        cycle_timing["capture_done_epoch_ms"] = epoch_ms()
        log_event("infer_begin", source=cmd.get("source", "manual"), cycle=cycle, roi=list(roi_pixels(frame, cfg)), cycle_timing=cycle_timing)
        result = inspect_frame(frame, model, labels, cfg, cmd.get("source", "manual"), cycle, cycle_timing=cycle_timing)
        return cap, {"ok": True, "result": result}
    if name == "inspect_file":
        img_path = Path(cmd.get("image_path", ""))
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise RuntimeError(f"Não abriu imagem: {img_path}")
        result = inspect_frame(frame, model, labels, cfg, "uploaded_file", cycle, cycle_timing={"command_epoch_ms": cmd.get("command_epoch_ms"), "command_seen_epoch_ms": epoch_ms(), "capture_begin_epoch_ms": epoch_ms(), "capture_done_epoch_ms": epoch_ms()})
        return cap, {"ok": True, "result": result}
    if name == "save_dataset":
        return cap, {"ok": True, **handle_dataset_copy(cmd)}
    if name == "generate_report":
        return cap, {"ok": True, **handle_report(cmd)}
    if name == "send_email":
        return cap, {"ok": True, **send_email(cmd)}
    if name == "cleanup_disk":
        return cap, {"ok": True, **cleanup_disk(cmd)}
    if name == "reset_summary":
        atomic_write_json(SUMMARY_PATH, {"timestamp": now(), "total": 0, "ok": 0, "ng": 0, "yield_pct": 0, "classes": {}})
        return cap, {"ok": True, "reset_summary": True}
    if name == "reset_sensor_state":
        log_event("sensor_state_reset_requested")
        return cap, {"ok": True, "reset_sensor_state": True}
    if name == "stop":
        log_event("stop_requested", request=cmd)
        return cap, {"ok": True, "stop": True}
    return cap, {"ok": False, "error": f"Comando desconhecido: {name}"}


def main():
    print("=" * 70)
    print("SVC USB v2.2 - CORE EXTERNO DE PRODUÇÃO + SCANNER TRIGGER")
    print("=" * 70)
    cfg = load_config()
    log_event("core_start", config=cfg)
    model, labels, model_path, model_info = load_model()
    cap = open_camera(cfg)
    ser = None
    serial_info = {"serial_state": "OFF", "serial_error": ""}
    sensor_state = {
        "raw_present": None,
        "stable_present": None,
        "prev_stable_present": None,
        "changed_at": time.time(),
        "last_fire_ts": 0.0,
        "last_sensor_fire": "---",
    }
    cycle = 0
    last_cmd_ts = None
    write_heartbeat("RUNNING", cycle, {"camera_index": cfg.get("camera_index"), "mode": "MANUAL/SCANNER/SENSOR", "model_path": model_path, "labels": labels, **model_info})
    try:
        while True:
            cfg = load_config()
            ser, serial_info = open_serial_if_needed(cfg, ser)
            hb_extra = {
                "camera_index": cfg.get("camera_index"),
                "mode": "SENSOR" if bool(cfg.get("serial_enabled", False)) else "MANUAL",
                "model_path": model_path,
                "labels": labels,
                "product_model_config_current": cfg.get("product_model"),
                "product_model_loaded": model_info.get("product_model_loaded", ""),
                "recipe_path": model_info.get("recipe_path", ""),
                "recipe_model_path": model_info.get("recipe_model_path", ""),
                "model_path_loaded": model_info.get("model_path_loaded", model_path),
                "model_source": model_info.get("model_source", ""),
                "serial_port": cfg.get("serial_port"),
                "serial_state": serial_info.get("serial_state", "OFF"),
                "serial_error": serial_info.get("serial_error", ""),
                "sensor_present": sensor_state.get("stable_present"),
                "sensor_raw_present": sensor_state.get("raw_present"),
                "sensor_loop": "main",
                "last_sensor_fire": sensor_state.get("last_sensor_fire", "---"),
                "auto_shift_reports_enabled": cfg.get("auto_shift_reports_enabled"),
                "auto_shift_email_enabled": cfg.get("auto_shift_email_enabled"),
                "shift_report_times": cfg.get("shift_report_times"),
            }
            write_heartbeat("RUNNING", cycle, hb_extra)

            cmd = read_json(COMMAND_PATH, None)
            if cmd and cmd.get("timestamp") != last_cmd_ts:
                last_cmd_ts = cmd.get("timestamp")
                try:
                    if cmd.get("command") in ("inspect_once", "test_once", "inspect_file"):
                        cycle += 1
                    if cmd.get("command") == "reset_sensor_state":
                        sensor_state.update({"raw_present": None, "stable_present": None, "prev_stable_present": None, "changed_at": time.time(), "last_fire_ts": 0.0, "last_sensor_fire": "reset"})
                    cap, ack = execute_command(cmd, cap, model, labels, cycle)
                    ack.update({"timestamp": now(), "command": cmd.get("command"), "cycle": cycle})
                    atomic_write_json(ACK_PATH, ack)
                    if ack.get("stop"):
                        break
                except Exception as e:
                    err = {"timestamp": now(), "command": cmd.get("command"), "cycle": cycle, "ok": False, "error": repr(e), "traceback": traceback.format_exc()}
                    atomic_write_json(ACK_PATH, err)
                    log_event("error", source=cmd.get("source", "command"), cycle=cycle, error=repr(e), traceback=traceback.format_exc())

            # Disparo automático por Arduino/sensor serial
            # O firmware envia PRESENT=0/1 apenas quando o estado muda.
            # Por isso, quando não chega linha nova, reavaliamos o último raw_present
            # para permitir que a estabilidade (serial_stable_ms) complete e gere o edge.
            line = read_sensor_line(ser)
            present = parse_sensor_present(line) if line is not None else None
            if line is not None:
                log_event("serial_line", line=line, present=present)

            present_for_fire = present if present is not None else sensor_state.get("raw_present")

            if bool(cfg.get("serial_enabled", False)) and maybe_fire_from_sensor(sensor_state, present_for_fire, cfg):
                try:
                    settle_s = max(0, int(cfg.get("sensor_settle_ms", 180))) / 1000.0
                    log_event("sensor_fire", cycle=cycle + 1, present=present_for_fire, settle_ms=int(settle_s * 1000), mode=cfg.get("trigger_mode"))
                    if settle_s:
                        time.sleep(settle_s)
                    cycle += 1
                    sensor_capture_begin_ms = epoch_ms()
                    frame, cap = safe_capture(cap, cfg)
                    sensor_capture_done_ms = epoch_ms()
                    inspect_frame(frame, model, labels, cfg, "sensor_serial", cycle, cycle_timing={"capture_begin_epoch_ms": sensor_capture_begin_ms, "capture_done_epoch_ms": sensor_capture_done_ms})
                except Exception as e:
                    err = {"timestamp": now(), "command": "sensor_serial", "cycle": cycle, "ok": False, "error": repr(e), "traceback": traceback.format_exc()}
                    atomic_write_json(ACK_PATH, err)
                    log_event("sensor_error", cycle=cycle, error=repr(e), traceback=traceback.format_exc())

            if bool(cfg.get("auto_trigger_enabled", False)):
                time.sleep(float(cfg.get("auto_trigger_interval_s", 2.0)))
                cycle += 1
                auto_capture_begin_ms = epoch_ms()
                frame, cap = safe_capture(cap, cfg)
                auto_capture_done_ms = epoch_ms()
                inspect_frame(frame, model, labels, cfg, "auto_timer", cycle, cycle_timing={"capture_begin_epoch_ms": auto_capture_begin_ms, "capture_done_epoch_ms": auto_capture_done_ms})

            # Relatórios automáticos ao final dos turnos configurados
            maybe_auto_shift_report(cfg)

            time.sleep(0.05)
    finally:
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        write_heartbeat("STOPPED", cycle)
        log_event("core_stopped", cycle=cycle)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        tb = traceback.format_exc()
        log_event("fatal_error", error=repr(e), traceback=tb)
        try:
            write_heartbeat("FATAL_ERROR", 0, {"error": repr(e)})
        except Exception:
            pass
        print(tb)
        raise
