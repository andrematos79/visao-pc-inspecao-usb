import json, time, csv, shutil, smtplib, ssl, mimetypes
from pathlib import Path
from datetime import datetime, date
from email.message import EmailMessage

APP_FOOTER_VERSION = "v2.1.0"
APP_RELEASE_STATUS = "Production"

import streamlit as st
import pandas as pd

import platform
import cv2
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
RUNTIME = BASE_DIR / "runtime_status"
LOG_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_DIR = BASE_DIR / "upload_tests"
CONFIG_PATH = BASE_DIR / "config_usb.json"
EMAIL_CONFIG_PATH = BASE_DIR / "config_email.json"
HEARTBEAT_PATH = RUNTIME / "heartbeat.json"
LAST_RESULT_PATH = RUNTIME / "last_result.json"
SUMMARY_PATH = RUNTIME / "summary.json"
COMMAND_PATH = RUNTIME / "command.json"
ACK_PATH = RUNTIME / "ack.json"
CSV_LOG_PATH = LOG_DIR / "inspection_log.csv"
IND_LOG_PATH = LOG_DIR / "industrial_log.jsonl"
CURRENT_CONTEXT_PATH = RUNTIME / "current_serial.json"
TRACE_LOG_PATH = LOG_DIR / "inspection_trace_log.csv"
MES_XML_DIR = LOG_DIR / "mes_xml"
APP_VERSION = "SVC USB - Computer Vision System for USB Inspection"
ENG_PIN_DEFAULT = "1234"
CLASSES = ["OK", "NG_DESALINHADO", "NG_DANIFICADO"]
NG_CLASSES = ["NG_DESALINHADO", "NG_DANIFICADO"]

for p in [RUNTIME, LOG_DIR, REPORTS_DIR, UPLOAD_DIR]:
    p.mkdir(exist_ok=True, parents=True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def read_json(path, default=None):
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def write_json(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config():
    return read_json(CONFIG_PATH, {}) or {}


def save_config(cfg):
    write_json(CONFIG_PATH, cfg)


def normalize_serial_qr(serial: str) -> str:
    import re
    s = (serial or "").strip().replace("+", "-")
    s = re.sub(r"\s+", "", s)
    return s


def current_context():
    ctx = read_json(CURRENT_CONTEXT_PATH, {}) or {}
    cfg = load_config()
    return {
        "serial_number": normalize_serial_qr(ctx.get("serial_number", "")),
        "production_order": str(ctx.get("production_order", cfg.get("production_order", ""))).strip(),
        "equipment_id": str(ctx.get("equipment_id", cfg.get("equipment_id", "SVC01"))).strip(),
        "line_name": str(ctx.get("line_name", cfg.get("line_name", "L01"))).strip(),
        "product_model": str(ctx.get("product_model", cfg.get("product_model", "UNDEFINED"))).strip(),
        "mes_enabled": bool(ctx.get("mes_enabled", cfg.get("mes_enabled", False))),
        "traceability_enabled": bool(ctx.get("traceability_enabled", cfg.get("traceability_enabled", False))),
        "timestamp": ctx.get("timestamp", now()),
    }


def save_context(ctx):
    payload = current_context()
    payload.update(ctx or {})
    payload["timestamp"] = now()
    payload["serial_number"] = normalize_serial_qr(payload.get("serial_number", ""))
    write_json(CURRENT_CONTEXT_PATH, payload)
    send_command("set_context", source="streamlit_mes_context", context=payload)
    return payload



def commit_serial_scan():
    """Callback do campo de scanner.
    Ao pressionar ENTER (scanner HID), salva o SN atual e limpa o campo visual,
    evitando concatenação do próximo QRCode no final do anterior.
    """
    scan = normalize_serial_qr(st.session_state.get("serial_scan_input", ""))
    if not scan:
        return
    ctx = current_context()
    ctx["serial_number"] = scan
    ctx["last_scan_at"] = now()
    save_context(ctx)
    st.session_state["last_scanned_serial"] = scan
    # Limpa o buffer visual do scanner para a próxima peça.
    st.session_state["serial_scan_input"] = ""


def inspection_block_reason(ctx, cfg):
    trace_on = bool(ctx.get("traceability_enabled")) or bool(ctx.get("mes_enabled")) or bool(cfg.get("traceability_enabled")) or bool(cfg.get("mes_enabled"))
    if not trace_on:
        return ""
    if len(normalize_serial_qr(ctx.get("serial_number", ""))) < int(cfg.get("serial_min_len", 4)):
        return "Escaneie o Número de Série / QRCode antes da inspeção."
    if not str(ctx.get("production_order", "")).strip():
        return "Informe a Ordem de Produção antes da inspeção."
    if not str(ctx.get("equipment_id", "")).strip():
        return "Informe o Equipment ID antes da inspeção."
    return ""


def send_command(command, **kwargs):
    payload = {"timestamp": now(), "command": command, **kwargs}
    write_json(COMMAND_PATH, payload)
    return payload


def heartbeat():
    return read_json(HEARTBEAT_PATH, {}) or {}


def last_result():
    return read_json(LAST_RESULT_PATH, {}) or {}


def summary():
    return read_json(SUMMARY_PATH, {"total": 0, "ok": 0, "ng": 0, "yield_pct": 0, "classes": {}}) or {}


def ack():
    return read_json(ACK_PATH, {}) or {}


def count_imgs(folder):
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"])


def load_csv_log():
    if not CSV_LOG_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CSV_LOG_PATH, sep=";", encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def filtered_log(period="today"):
    df = load_csv_log()
    if df.empty or "timestamp" not in df.columns:
        return df
    df["_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if period == "today":
        today = pd.Timestamp(date.today())
        df = df[df["_dt"] >= today]
    return df


def compute_stats_from_df(df):
    if df.empty:
        return {"total": 0, "ok": 0, "ng": 0, "yield_pct": 0, "classes": {}}
    total = len(df)
    ok = int((df["class_name"] == "OK").sum()) if "class_name" in df.columns else 0
    ng = total - ok
    classes = df["class_name"].value_counts().to_dict() if "class_name" in df.columns else {}
    return {"total": total, "ok": ok, "ng": ng, "yield_pct": round(100 * ok / max(total, 1), 2), "classes": classes}


def render_big_status(res):
    cls = res.get("class_name") or "---"
    conf = float(res.get("confidence") or 0)
    if cls == "OK":
        st.markdown("""
        <div style='background:#d8f5dd;border:3px solid #19a83a;border-radius:22px;padding:34px;text-align:center'>
        <div style='font-size:64px;font-weight:900;color:#087c22'>OK</div>
        <div style='font-size:20px;color:#075c1b'>Produto aprovado</div>
        </div>
        """, unsafe_allow_html=True)
    elif cls.startswith("NG"):
        st.markdown("""
        <div style='background:#ffe0e0;border:3px solid #d71920;border-radius:22px;padding:34px;text-align:center'>
        <div style='font-size:64px;font-weight:900;color:#b00000'>REPROVADO</div>
        <div style='font-size:20px;color:#800000'>Produto bloqueado para análise</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#f2f2f2;border:3px solid #999;border-radius:22px;padding:34px;text-align:center'>
        <div style='font-size:48px;font-weight:900;color:#555'>AGUARDANDO</div>
        </div>
        """, unsafe_allow_html=True)
    if cls != "---":
        st.caption(f"Causa provável interna: {cls} | confiança: {conf:.3f} | ciclo: {res.get('cycle','-')}")


def render_audit_charts(df):
    if df.empty or "class_name" not in df.columns:
        st.info("Sem dados suficientes para gráficos.")
        return
    classes = df["class_name"].value_counts().rename_axis("Classe").reset_index(name="Quantidade")
    st.bar_chart(classes.set_index("Classe"))
    ng = df[df["class_name"].astype(str).str.startswith("NG")]
    if not ng.empty:
        st.markdown("#### Causas prováveis entre reprovados")
        ng_counts = ng["class_name"].value_counts()
        total_ng = max(int(ng_counts.sum()), 1)
        table = pd.DataFrame({"Classe": ng_counts.index, "Quantidade": ng_counts.values})
        table["Percentual_NG"] = (100 * table["Quantidade"] / total_ng).round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)


def latest_reports():
    return sorted(REPORTS_DIR.glob("relatorio_svc_usb_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)


def send_email_direct(to, subject, body, attachment=None):
    cfg = read_json(EMAIL_CONFIG_PATH, {}) or {}
    server = cfg.get("smtp_server", "smtp.office365.com")
    port = int(cfg.get("smtp_port", 587))
    user = cfg.get("smtp_user", "")
    pwd = cfg.get("smtp_password", "")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)
    if attachment and Path(attachment).exists():
        path = Path(attachment)
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)
    with smtplib.SMTP(server, port, timeout=30) as s:
        if cfg.get("smtp_use_tls", True):
            s.starttls(context=ssl.create_default_context())
        if user:
            s.login(user, pwd)
        s.send_message(msg)


st.set_page_config(page_title="SVC USB v2.0", layout="wide")
st.title(APP_VERSION)
st.caption("Operação: OK/REPROVADO. Auditoria: causa provável detalhada por classe NG.")

if "eng_unlocked" not in st.session_state:
    st.session_state.eng_unlocked = False

cfg = load_config()
eng_pin = str(cfg.get("eng_pin", ENG_PIN_DEFAULT))

with st.sidebar:
  
    logo_candidates = [BASE_DIR / "assets" / "logo_sistema.png", BASE_DIR / "assets" / "logo_svc_usb.png", BASE_DIR / "logo_svc.png"]
    logo_path = next((p for p in logo_candidates if p.exists()), None)
    if logo_path:
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown("""
        <div style='background:white;border-radius:14px;padding:14px;text-align:center;border:1px solid #ddd;margin-bottom:12px'>
          <div style='font-size:34px;font-weight:900;color:#1c4f91'>SVC USB</div>
          <div style='font-size:12px;color:#666'>Computer Vision System</div>
        </div>
        """, unsafe_allow_html=True)
    with st.expander("ℹ️ Sobre o Sistema", expanded=False):
        st.markdown("### Sistema: SVC USB – Computer Vision System for USB Inspection")
        st.markdown("- **Versão:** v2.1.0 - Production Validation")
        st.markdown("- **Status:** Pré-validação industrial com sensor")
        st.markdown("- **Release:** 21/05/2026")
        st.markdown("- **Aplicação:** Automated Visual Inspection of USB Connectors")
        st.markdown("- **Desenvolvedor:** André Gama de Matos")
        st.markdown("- **Orientador Acadêmico:** Prof. Lucas Delapria Dias dos Santos")
        st.markdown("- **Curso:** Engenharia de Controle e Automação")
        st.markdown("- **Instituição:** Centro Universitário Unifatecie")
        st.markdown("- **Projeto:** Trabalho de Conclusão de Curso - TCC")
        st.markdown("- **Licença:** MIT License")
        st.markdown("---")
        st.markdown("**Ambiente de Execução do Sistema**")
        st.markdown(f"- **Sistema Operacional:** {platform.system()} {platform.release()}")
        st.markdown(f"- **Python:** {platform.python_version()}")
        st.markdown(f"- **OpenCV:** {cv2.__version__}")
        st.markdown(f"- **TensorFlow:** {tf.__version__}")
        st.markdown("---")
        st.markdown("**Arquitetura do Sistema**")
        st.markdown(f"- **Aquisição de imagem:** Microscópio digital USB")
        st.markdown(f"- **Processamento de imagem:** OpenCV")
        st.markdown(f"- **Modelo de IA:** CNN MobileNetV2")
        st.markdown(f"- **Framework de inferência:** TensorFlow/Keras")
        st.markdown("- **Interface de usuário:** Streamlit")
    

    st.header("Controle de Linha")
    auto = st.checkbox("Auto-refresh", value=True)
    interval = st.slider("Refresh (ms)", 500, 5000, 1000, 100)

    st.divider()
    st.subheader("🏭 Produção / MES")
    ctx0 = current_context()
    mes_enabled = st.checkbox("Ativar MES", value=bool(cfg.get("mes_enabled", ctx0.get("mes_enabled", False))))
    traceability_enabled = st.checkbox("Ativar rastreabilidade por Serial / QRCode", value=bool(cfg.get("traceability_enabled", ctx0.get("traceability_enabled", False))))
    production_order = st.text_input("Ordem de Produção", value=str(ctx0.get("production_order", cfg.get("production_order", ""))), placeholder="Ex.: BK4338BRI_Y25")
    equipment_id = st.text_input("Equipment ID", value=str(ctx0.get("equipment_id", cfg.get("equipment_id", "SVC01"))))
    line_name = st.text_input("Linha", value=str(ctx0.get("line_name", cfg.get("line_name", "L01"))))
    product_model = st.text_input("Modelo", value=str(ctx0.get("product_model", cfg.get("product_model", "UNDEFINED"))))
    if "serial_scan_input" not in st.session_state:
        st.session_state["serial_scan_input"] = ""
    st.text_input(
        "Número de Série / QRCode",
        key="serial_scan_input",
        placeholder="Passe o scanner aqui e pressione ENTER",
        on_change=commit_serial_scan,
    )
    ctx_after_scan = current_context()
    ctx = save_context({
        "mes_enabled": bool(mes_enabled),
        "traceability_enabled": bool(traceability_enabled),
        "production_order": production_order,
        "equipment_id": equipment_id,
        "line_name": line_name,
        "product_model": product_model,
        # O campo visual é apenas buffer do scanner. O serial válido fica salvo no contexto.
        "serial_number": ctx_after_scan.get("serial_number", ""),
    })
    cfg.update({
        "mes_enabled": bool(mes_enabled),
        "traceability_enabled": bool(traceability_enabled),
        "production_order": production_order,
        "equipment_id": equipment_id,
        "line_name": line_name,
        "product_model": product_model,
    })
    save_config(cfg)
    st.caption(f"MES: {'ATIVO' if mes_enabled else 'DESLIGADO'} | Rastreabilidade: {'ATIVA' if traceability_enabled else 'DESLIGADA'}")
    st.caption(f"Serial atual: {ctx.get('serial_number') or '---'}")
    if st.button("🧹 Limpar serial atual", use_container_width=True):
        ctx = save_context({"serial_number": ""})
        send_command("clear_serial", source="streamlit_mes_clear")
        st.success("Serial atual limpo.")
    st.divider()
    block_reason = inspection_block_reason(current_context(), cfg)
    if st.button("🔎 INSPECIONAR 1 PEÇA", use_container_width=True, type="primary", disabled=bool(block_reason)):
        send_command("inspect_once", source="operator_manual_button")
        st.success("Comando enviado ao core.")
    if block_reason:
        st.warning(block_reason)
    if st.button("📄 Gerar relatório agora", use_container_width=True):
        send_command("generate_report", source="streamlit_sidebar", period="today")
        st.success("Comando de relatório enviado.")
    if st.button("🔄 Zerar contadores", use_container_width=True):
        send_command("reset_summary", source="streamlit_sidebar")
        st.success("Comando enviado ao core.")
    if st.button("🛑 Parar core", use_container_width=True):
        send_command("stop", source="streamlit_sidebar")
        st.warning("Comando de parada enviado.")
    st.divider()
    pin = st.text_input("PIN Engenharia", type="password")
    if st.button("Liberar Engenharia", use_container_width=True):
        st.session_state.eng_unlocked = (pin == eng_pin)
        if not st.session_state.eng_unlocked:
            st.warning("PIN inválido.")
    if st.session_state.eng_unlocked:
        st.success("Modo Engenharia liberado")
    else:
        st.info("Modo Operador")

    st.divider()
    st.subheader("🔌 Debug Serial / Sensor")
    serial_enabled = st.checkbox("Sensor serial habilitado", value=bool(cfg.get("serial_enabled", False)), disabled=not st.session_state.eng_unlocked)
    serial_port = st.text_input("Porta Arduino", value=str(cfg.get("serial_port", "COM1")), disabled=not st.session_state.eng_unlocked)
    serial_baud = st.selectbox("Baud", [9600, 57600, 115200], index=[9600, 57600, 115200].index(int(cfg.get("serial_baud", 115200))) if int(cfg.get("serial_baud", 115200)) in [9600,57600,115200] else 2, disabled=not st.session_state.eng_unlocked)
    trigger_mode = st.selectbox("Trigger mode", ["edge_0to1", "stable_high"], index=0 if str(cfg.get("trigger_mode", "edge_0to1")) == "edge_0to1" else 1, disabled=not st.session_state.eng_unlocked)
    sensor_settle_ms = st.number_input("Settle após sensor=1 (ms)", 0, 3000, int(cfg.get("sensor_settle_ms", 180)), 10, disabled=not st.session_state.eng_unlocked)
    serial_stable_ms = st.number_input("Estabilidade serial (ms)", 0, 2000, int(cfg.get("serial_stable_ms", 80)), 10, disabled=not st.session_state.eng_unlocked)
    serial_rearm_ms = st.number_input("Rearme após disparo (ms)", 0, 5000, int(cfg.get("serial_rearm_ms", 600)), 50, disabled=not st.session_state.eng_unlocked)
    if st.button("Salvar config do sensor", disabled=not st.session_state.eng_unlocked, use_container_width=True):
        cfg.update({
            "serial_enabled": bool(serial_enabled),
            "serial_port": serial_port,
            "serial_baud": int(serial_baud),
            "trigger_mode": trigger_mode,
            "sensor_settle_ms": int(sensor_settle_ms),
            "serial_stable_ms": int(serial_stable_ms),
            "serial_rearm_ms": int(serial_rearm_ms),
        })
        save_config(cfg)
        send_command("reset_sensor_state", source="streamlit_sensor_config")
        st.success("Config do sensor salva. Reinicie o core se a porta COM mudou.")

hb = heartbeat()
res = last_result()
sm = summary()
ack_data = ack()
status = hb.get("state", "OFFLINE")
age = "---"
try:
    t = datetime.strptime(hb.get("timestamp", ""), "%Y-%m-%d %H:%M:%S.%f")
    age = f"{(datetime.now() - t).total_seconds():.1f}s"
except Exception:
    pass

with st.sidebar.expander("Estado do sensor", expanded=True):
    st.write("Serial:", hb.get("serial_state", "OFF"))
    st.write("Porta:", hb.get("serial_port", cfg.get("serial_port", "-")))
    st.write("Último PRESENT:", hb.get("sensor_present", "---"))
    st.write("Thread/loop:", hb.get("sensor_loop", "main"))
    st.write("Último disparo:", hb.get("last_sensor_fire", "---"))
    st.write("Erro:", hb.get("serial_error", ""))

with st.sidebar.expander("MES / Rastreabilidade", expanded=True):
    ctx_view = current_context()
    st.write("MES:", "ATIVO" if ctx_view.get("mes_enabled") else "DESLIGADO")
    st.write("Rastreabilidade:", "ATIVA" if ctx_view.get("traceability_enabled") else "DESLIGADA")
    st.write("OP:", ctx_view.get("production_order") or "---")
    st.write("Equipamento:", ctx_view.get("equipment_id") or "---")
    st.write("Linha:", ctx_view.get("line_name") or "---")
    st.write("Modelo:", ctx_view.get("product_model") or "---")
    st.write("Serial atual:", ctx_view.get("serial_number") or "---")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Core", status)
k2.metric("Heartbeat", age)
k3.metric("Total", sm.get("total", 0))
k4.metric("OK", sm.get("ok", 0))
k5.metric("NG", sm.get("ng", 0))
k6.metric("Yield", f"{sm.get('yield_pct', 0)}%")
ctx_top = current_context()
st.caption(f"OP: {ctx_top.get('production_order') or '---'} | Serial atual: {ctx_top.get('serial_number') or '---'} | Equip.: {ctx_top.get('equipment_id') or '---'} | MES: {'ATIVO' if ctx_top.get('mes_enabled') else 'DESLIGADO'} | Rastreab.: {'ATIVA' if ctx_top.get('traceability_enabled') else 'DESLIGADA'}")

left, right = st.columns([0.95, 1.05])
with left:
    st.subheader("Resultado do Operador")
    render_big_status(res)
    if st.session_state.eng_unlocked:
        probs = res.get("probs") or res.get("merged_probs") or {}
        if probs:
            st.markdown("#### Probabilidades internas")
            rows = [{"Classe": k, "Probabilidade": round(float(v), 4)} for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)]
            st.dataframe(rows, use_container_width=True, hide_index=True)
with right:
    st.subheader("Imagem / ROI")
    overlay_path = res.get("overlay_path") or str(RUNTIME / "last_frame_overlay.jpg")
    if Path(overlay_path).exists():
        st.image(overlay_path, caption="Frame com ROI USB", use_container_width=True)
    else:
        st.info("Aguardando primeira inspeção.")

st.divider()
tabs = st.tabs(["Auditoria", "Teste Engenharia", "Dataset", "Relatórios/E-mail", "Espaço em Disco", "MES/Rastreabilidade", "Config/Logs"])

with tabs[0]:
    st.subheader("Auditoria de Produção")
    period = st.radio("Período", ["today", "all"], format_func=lambda x: "Hoje" if x == "today" else "Tudo", horizontal=True)
    df = filtered_log(period)
    stats = compute_stats_from_df(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total período", stats["total"])
    c2.metric("OK período", stats["ok"])
    c3.metric("NG período", stats["ng"])
    c4.metric("Yield período", f"{stats['yield_pct']}%")
    render_audit_charts(df)
    if not df.empty:
        with st.expander("Últimas inspeções", expanded=False):
            st.dataframe(df.tail(200).drop(columns=["_dt"], errors="ignore"), use_container_width=True)

with tabs[1]:
    st.subheader("Modo Engenharia — teste sem câmera")
    if not st.session_state.eng_unlocked:
        st.warning("Libere Engenharia na barra lateral.")
    up = st.file_uploader("Carregar foto JPG/PNG/BMP", type=["jpg", "jpeg", "png", "bmp"], disabled=not st.session_state.eng_unlocked)
    if up is not None:
        dst = UPLOAD_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{up.name}"
        dst.write_bytes(up.getbuffer())
        st.image(str(dst), caption="Imagem carregada", use_container_width=True)
        if st.button("Inferir imagem carregada pelo core", disabled=not st.session_state.eng_unlocked):
            send_command("inspect_file", image_path=str(dst), source="engineering_upload")
            st.success("Comando enviado ao core.")

with tabs[2]:
    st.subheader("Coleta / Curadoria de Dataset")
    dataset_dir = BASE_DIR / str(cfg.get("dataset_dir", "dataset_usb_live_capture"))
    st.caption(f"Pasta de captura/dataset operacional: {dataset_dir}")
    cols = st.columns(3)
    for col, cls in zip(cols, CLASSES):
        with col:
            st.metric(cls, count_imgs(dataset_dir / cls))
            if st.button(f"Salvar último ROI como {cls}", key=f"save_{cls}", disabled=not st.session_state.eng_unlocked, use_container_width=True):
                send_command("save_dataset", class_name=cls, source="streamlit_dataset_button")
                st.success(f"Comando enviado: {cls}")
    st.info("Recomendação: use esta pasta como captura operacional. Promova manualmente imagens boas para dataset_usb_v15.")

with tabs[3]:
    st.subheader("Relatórios de Auditoria e E-mail")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Gerar relatório HTML/PDF", use_container_width=True):
            send_command("generate_report", source="streamlit_reports", period="today")
            st.success("Comando enviado ao core.")
        reps = latest_reports()
        if reps:
            st.write("Últimos relatórios:")
            for r in reps[:6]:
                st.download_button(f"Baixar {r.name}", data=r.read_bytes(), file_name=r.name, mime="application/octet-stream")
    with col_b:
        cfg_email = read_json(EMAIL_CONFIG_PATH, {}) or {}
        smtp_server = st.text_input("SMTP server", cfg_email.get("smtp_server", "smtp.office365.com"), disabled=not st.session_state.eng_unlocked)
        smtp_port = st.number_input("SMTP port", value=int(cfg_email.get("smtp_port", 587)), step=1, disabled=not st.session_state.eng_unlocked)
        smtp_user = st.text_input("Usuário SMTP", cfg_email.get("smtp_user", ""), disabled=not st.session_state.eng_unlocked)
        smtp_password = st.text_input("Senha SMTP", cfg_email.get("smtp_password", ""), type="password", disabled=not st.session_state.eng_unlocked)
        email_to = st.text_input("Destinatário(s)", cfg_email.get("to", ""), disabled=not st.session_state.eng_unlocked)
        if st.button("Salvar config e-mail", disabled=not st.session_state.eng_unlocked):
            write_json(EMAIL_CONFIG_PATH, {"smtp_server": smtp_server, "smtp_port": smtp_port, "smtp_user": smtp_user, "smtp_password": smtp_password, "smtp_use_tls": True, "to": email_to, "subject": "Relatório SVC USB"})
            st.success("Configuração salva.")
        if st.button("Enviar último relatório", disabled=not st.session_state.eng_unlocked):
            reps = latest_reports()
            attach = str(reps[0]) if reps else ""
            send_command("send_email", to=email_to, attachment=attach, subject="Relatório SVC USB", body="Segue relatório do SVC USB.")
            st.success("Comando enviado ao core.")

with tabs[4]:
    st.subheader("Controle de Espaço em Disco")
    usage = shutil.disk_usage(BASE_DIR)
    st.metric("Disco livre", f"{usage.free/1024**3:.1f} GB")
    st.metric("Uso da pasta captures", f"{sum(f.stat().st_size for f in (BASE_DIR/str(cfg.get('captures_dir','captures_usb'))).rglob('*') if f.is_file())/1024**2:.1f} MB" if (BASE_DIR/str(cfg.get('captures_dir','captures_usb'))).exists() else "0 MB")
    keep_days = st.number_input("Manter imagens/logs por quantos dias", value=int(cfg.get("retention_days", 30)), step=1, disabled=not st.session_state.eng_unlocked)
    if st.button("Executar limpeza automática", disabled=not st.session_state.eng_unlocked):
        send_command("cleanup_disk", source="streamlit_disk", keep_days=int(keep_days))
        st.success("Comando enviado ao core.")

with tabs[5]:
    st.subheader("MES / Rastreabilidade")
    ctx_mes = current_context()
    st.json(ctx_mes, expanded=True)
    if TRACE_LOG_PATH.exists():
        try:
            df_trace = pd.read_csv(TRACE_LOG_PATH, sep=";", encoding="utf-8")
            st.markdown("#### Últimos registros de rastreabilidade")
            st.dataframe(df_trace.tail(200), use_container_width=True)
        except Exception as e:
            st.warning(f"Falha ao ler inspection_trace_log.csv: {e}")
    else:
        st.info("Ainda não existe inspection_trace_log.csv.")
    if MES_XML_DIR.exists():
        xmls = sorted(MES_XML_DIR.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
        st.markdown("#### Últimos XMLs MES locais")
        for x in xmls[:10]:
            st.download_button(f"Baixar {x.name}", data=x.read_bytes(), file_name=x.name, mime="application/xml")

with tabs[6]:
    st.subheader("Configuração / Logs")
    if not st.session_state.eng_unlocked:
        st.warning("Libere Engenharia para alterar configurações.")
    with st.expander("Config USB", expanded=True):
        camera_index = st.number_input("camera_index", value=int(cfg.get("camera_index", 0)), step=1, disabled=not st.session_state.eng_unlocked)
        camera_backend = st.selectbox("camera_backend", ["auto", "dshow", "msmf"], index=["auto", "dshow", "msmf"].index(str(cfg.get("camera_backend", "auto"))) if str(cfg.get("camera_backend", "auto")) in ["auto", "dshow", "msmf"] else 0, disabled=not st.session_state.eng_unlocked)
        rx0 = st.number_input("roi_x0", 0.0, 1.0, float(cfg.get("roi_x0", 0.0)), 0.01, disabled=not st.session_state.eng_unlocked)
        ry0 = st.number_input("roi_y0", 0.0, 1.0, float(cfg.get("roi_y0", 0.0)), 0.01, disabled=not st.session_state.eng_unlocked)
        rx1 = st.number_input("roi_x1", 0.0, 1.0, float(cfg.get("roi_x1", 1.0)), 0.01, disabled=not st.session_state.eng_unlocked)
        ry1 = st.number_input("roi_y1", 0.0, 1.0, float(cfg.get("roi_y1", 1.0)), 0.01, disabled=not st.session_state.eng_unlocked)
        save_all = st.checkbox("Salvar todas as capturas", value=bool(cfg.get("save_all_captures", False)), disabled=not st.session_state.eng_unlocked)
        save_ng = st.checkbox("Salvar NG automaticamente", value=bool(cfg.get("save_ng_images", True)), disabled=not st.session_state.eng_unlocked)
        block_without_serial = st.checkbox("Bloquear inspeção sem serial quando rastreabilidade ativa", value=bool(cfg.get("block_without_serial", True)), disabled=not st.session_state.eng_unlocked)
        clear_after = st.checkbox("Limpar serial após cada inspeção", value=bool(cfg.get("clear_serial_after_inspection", True)), disabled=not st.session_state.eng_unlocked)
        mes_xml_enabled = st.checkbox("Gerar XML MES local", value=bool(cfg.get("mes_xml_enabled", True)), disabled=not st.session_state.eng_unlocked)
        if st.button("Salvar config_usb.json", disabled=not st.session_state.eng_unlocked):
            cfg.update({"camera_index": camera_index, "camera_backend": camera_backend, "roi_x0": rx0, "roi_y0": ry0, "roi_x1": rx1, "roi_y1": ry1, "save_all_captures": save_all, "save_ng_images": save_ng, "retention_days": int(keep_days), "block_without_serial": bool(block_without_serial), "clear_serial_after_inspection": bool(clear_after), "mes_xml_enabled": bool(mes_xml_enabled)})
            save_config(cfg)
            st.success("Config salva. Reinicie o core para garantir aplicação completa.")
    st.markdown("#### ACK")
    st.json(ack_data, expanded=False)
    st.markdown("#### Heartbeat")
    st.json(hb, expanded=False)
    st.markdown("#### Summary")
    st.json(sm, expanded=False)
    if IND_LOG_PATH.exists():
        st.markdown("#### LOG industrial expandido")
        lines = IND_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
        st.code("\n".join(lines), language="json")

st.markdown(
    f"""
    <div style="
        position: fixed;
        right: 22px;
        bottom: 8px;
        color: rgba(90, 100, 115, 0.45);
        font-size: 13px;
        font-weight: 500;
        z-index: 999999;
        pointer-events: none;
    ">
        {APP_FOOTER_VERSION} - Developed by André Gama de Matos - Software Engineer
    </div>
    """,
    unsafe_allow_html=True
)

if auto:
    time.sleep(interval / 1000.0)
    st.rerun()
