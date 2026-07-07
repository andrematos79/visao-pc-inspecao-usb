from pathlib import Path
import json
import csv
import random
import shutil
import numpy as np
import tensorflow as tf

# ============================================================
# TREINO SVC USB - NAPLES_BLACK v0.1 / NPI
# FIX: split estratificado por classe para matriz de confusao valida
#
# Dataset: datasets/NAPLES_BLACK_AUG200
# Saida : models/NAPLES_BLACK/naples_black_v01.keras
# ============================================================

SCRIPT_VERSION = "NAPLES_BLACK_v01_STRATIFIED_SPLIT_FIX_2026_07_07"

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "datasets" / "NAPLES_BLACK_AUG200"
OUTPUT_DIR = ROOT_DIR / "models" / "NAPLES_BLACK"

MODEL_PATH = OUTPUT_DIR / "naples_black_v01.keras"
BEST_HEAD_MODEL_PATH = OUTPUT_DIR / "naples_black_v01_best_head.keras"
BEST_FINE_MODEL_PATH = OUTPUT_DIR / "naples_black_v01_best_fine.keras"
BEST_MODEL_PATH = OUTPUT_DIR / "naples_black_v01_best.keras"

HISTORY_CSV = OUTPUT_DIR / "history_naples_black_v01.csv"
METRICS_TXT = OUTPUT_DIR / "metrics_naples_black_v01.txt"
CONFUSION_CSV = OUTPUT_DIR / "confusion_matrix_naples_black_v01.csv"
CLASS_NAMES_JSON = OUTPUT_DIR / "class_names_naples_black_v01.json"
SPLIT_CSV = OUTPUT_DIR / "split_naples_black_v01.csv"

# IMPORTANTE: manter esta ordem alinhada com o core/receita do SVC USB.
CLASSES = ["OK", "NG_DESALINHADO", "NG_DANIFICADO"]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.20

EPOCHS_HEAD = 25
EPOCHS_FINE = 10
FINE_TUNE = True
FINE_TUNE_LAST_N_LAYERS = 30

IMG_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def list_images(folder: Path):
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def count_images_by_class(dataset_dir: Path):
    counts = {}
    for cls in CLASSES:
        folder = dataset_dir / cls
        if not folder.exists():
            raise FileNotFoundError(f"Pasta de classe nao encontrada: {folder}")
        counts[cls] = len(list_images(folder))
    return counts


def clean_previous_outputs():
    """Remove arquivos de saida antigos para evitar confundir treino novo com checkpoint antigo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files_to_remove = [
        MODEL_PATH,
        BEST_HEAD_MODEL_PATH,
        BEST_FINE_MODEL_PATH,
        BEST_MODEL_PATH,
        HISTORY_CSV,
        METRICS_TXT,
        CONFUSION_CSV,
        CLASS_NAMES_JSON,
        SPLIT_CSV,
        OUTPUT_DIR / "history_naples_black_v01_head.csv",
        OUTPUT_DIR / "history_naples_black_v01_fine.csv",
    ]

    for path in files_to_remove:
        if path.exists():
            path.unlink()


def make_stratified_split():
    """
    Cria split 80/20 por classe.
    Com 200 imagens/classe, fica:
      treino:    160 por classe
      validacao:  40 por classe

    Isto evita o erro do validation_split automatico pegar validacao concentrada
    em uma unica classe.
    """
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []

    split_rows = []

    for class_index, class_name in enumerate(CLASSES):
        class_dir = DATASET_DIR / class_name
        images = list_images(class_dir)

        if not images:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {class_dir}")

        rng = random.Random(SEED + class_index)
        images = images[:]
        rng.shuffle(images)

        n_total = len(images)
        n_val = int(round(n_total * VAL_SPLIT))
        n_val = max(1, n_val)
        n_val = min(n_val, n_total - 1) if n_total > 1 else 1

        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for p in train_imgs:
            train_paths.append(str(p))
            train_labels.append(class_index)
            split_rows.append({
                "subset": "train",
                "class": class_name,
                "label": class_index,
                "filename": p.name,
                "path": str(p),
            })

        for p in val_imgs:
            val_paths.append(str(p))
            val_labels.append(class_index)
            split_rows.append({
                "subset": "val",
                "class": class_name,
                "label": class_index,
                "filename": p.name,
                "path": str(p),
            })

    # Embaralha treino. Validacao pode ser embaralhada tambem, mas nao precisa.
    train_pairs = list(zip(train_paths, train_labels))
    rng_train = random.Random(SEED)
    rng_train.shuffle(train_pairs)
    train_paths, train_labels = zip(*train_pairs)

    val_pairs = list(zip(val_paths, val_labels))
    rng_val = random.Random(SEED + 999)
    rng_val.shuffle(val_pairs)
    val_paths, val_labels = zip(*val_pairs)

    return list(train_paths), list(train_labels), list(val_paths), list(val_labels), split_rows


def save_split_csv(split_rows):
    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["subset", "class", "label", "filename", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)


def summarize_split(labels, title):
    print(title)
    labels = list(labels)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:16s}: {labels.count(i)}")
    print(f"  {'TOTAL':16s}: {len(labels)}")
    print()


def load_and_preprocess_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    # Manter escala 0..255 para MobileNetV2 preprocess_input.
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")

    label = tf.one_hot(label, depth=len(CLASSES))
    return img, label


def make_tf_dataset(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets():
    train_paths, train_labels, val_paths, val_labels, split_rows = make_stratified_split()
    save_split_csv(split_rows)

    print("Split estratificado confirmado:")
    summarize_split(train_labels, "Treino:")
    summarize_split(val_labels, "Validacao:")

    train_ds = make_tf_dataset(train_paths, train_labels, training=True)
    val_ds = make_tf_dataset(val_paths, val_labels, training=False)

    return train_ds, val_ds, train_paths, train_labels, val_paths, val_labels


def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="usb_roi_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
        )
        print("MobileNetV2 carregada com pesos ImageNet.")
    except Exception as exc:
        print("AVISO: nao foi possivel carregar pesos ImageNet.")
        print(f"Motivo: {exc}")
        print("Continuando com weights=None.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights=None,
        )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="classificador_usb")(x)

    model = tf.keras.Model(inputs, outputs, name="SVC_USB_NAPLES_BLACK_v01")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def make_callbacks(stage_name: str, checkpoint_path: Path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / f"history_naples_black_v01_{stage_name}.csv"),
            append=False,
        ),
    ]


def append_history_csv(histories):
    rows = []

    for stage, history in histories:
        epochs = len(history.history.get("loss", []))
        for i in range(epochs):
            row = {"stage": stage, "epoch": i + 1}
            for key, values in history.history.items():
                row[key] = values[i]
            rows.append(row)

    if not rows:
        return

    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compute_confusion_matrix(model, val_ds):
    y_true = []
    y_pred = []

    for batch_x, batch_y in val_ds:
        pred = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(pred, axis=1).tolist())

    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def save_confusion_matrix(cm):
    with open(CONFUSION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Real\\Predito"] + CLASSES)
        for i, cls in enumerate(CLASSES):
            writer.writerow([cls] + cm[i].tolist())


def classification_metrics_from_cm(cm):
    metrics = {}
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total else 0.0

    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }

    macro_precision = float(np.mean([m["precision"] for m in metrics.values()]))
    macro_recall = float(np.mean([m["recall"] for m in metrics.values()]))
    macro_f1 = float(np.mean([m["f1"] for m in metrics.values()]))

    return accuracy, macro_precision, macro_recall, macro_f1, metrics


def save_metrics_txt(counts, eval_loss, eval_acc, cm, chosen_model_name, train_labels, val_labels):
    accuracy, macro_precision, macro_recall, macro_f1, per_class = classification_metrics_from_cm(cm)

    lines = []
    lines.append("SVC USB - NAPLES_BLACK v0.1 / NPI")
    lines.append("====================================")
    lines.append(f"Script version: {SCRIPT_VERSION}")
    lines.append(f"Dataset: {DATASET_DIR}")
    lines.append(f"Modelo salvo: {MODEL_PATH}")
    lines.append(f"Melhor modelo usado: {chosen_model_name}")
    lines.append("")
    lines.append("Contagem total por classe:")
    for cls in CLASSES:
        lines.append(f"- {cls}: {counts[cls]}")
    lines.append("")
    lines.append("Split estratificado:")
    for i, cls in enumerate(CLASSES):
        lines.append(
            f"- {cls}: treino={list(train_labels).count(i)}, validacao={list(val_labels).count(i)}"
        )
    lines.append("")
    lines.append(f"Evaluation loss: {eval_loss:.6f}")
    lines.append(f"Evaluation accuracy: {eval_acc:.6f}")
    lines.append(f"CM accuracy: {accuracy:.6f}")
    lines.append(f"Macro precision: {macro_precision:.6f}")
    lines.append(f"Macro recall: {macro_recall:.6f}")
    lines.append(f"Macro F1: {macro_f1:.6f}")
    lines.append("")
    lines.append("Metricas por classe:")
    for cls, m in per_class.items():
        lines.append(
            f"- {cls}: precision={m['precision']:.6f}, recall={m['recall']:.6f}, "
            f"f1={m['f1']:.6f}, support={m['support']}"
        )
    lines.append("")
    lines.append("Matriz de confusao - linhas=real, colunas=predito:")
    lines.append(";".join(["Real\\Predito"] + CLASSES))
    for i, cls in enumerate(CLASSES):
        lines.append(";".join([cls] + [str(v) for v in cm[i].tolist()]))
    lines.append("")
    lines.append("Observacao: dataset NAPLES_BLACK_AUG200 e adequado para treino inicial/NPI.")
    lines.append("A classe NG_DESALINHADO nasceu de apenas 17 imagens reais e deve ser reforcada em v0.2.")

    METRICS_TXT.write_text("\n".join(lines), encoding="utf-8")


def choose_best_model(current_model, val_ds):
    candidates = []

    # Modelo em memoria apos o ultimo fit.
    current_loss, current_acc = current_model.evaluate(val_ds, verbose=0)
    candidates.append(("current_memory", None, current_loss, current_acc, current_model))

    # Checkpoint do head.
    if BEST_HEAD_MODEL_PATH.exists():
        head_model = tf.keras.models.load_model(BEST_HEAD_MODEL_PATH)
        head_loss, head_acc = head_model.evaluate(val_ds, verbose=0)
        candidates.append(("best_head", BEST_HEAD_MODEL_PATH, head_loss, head_acc, head_model))

    # Checkpoint do fine.
    if BEST_FINE_MODEL_PATH.exists():
        fine_model = tf.keras.models.load_model(BEST_FINE_MODEL_PATH)
        fine_loss, fine_acc = fine_model.evaluate(val_ds, verbose=0)
        candidates.append(("best_fine", BEST_FINE_MODEL_PATH, fine_loss, fine_acc, fine_model))

    candidates.sort(key=lambda item: (item[3], -item[2]), reverse=True)

    print("Comparacao de checkpoints:")
    for name, path, loss, acc, _ in candidates:
        print(f"  {name:14s}: accuracy={acc:.6f} | loss={loss:.6f} | path={path}")
    print()

    chosen_name, chosen_path, chosen_loss, chosen_acc, chosen_model = candidates[0]

    chosen_model.save(MODEL_PATH)
    shutil.copy2(MODEL_PATH, BEST_MODEL_PATH)

    return chosen_model, chosen_name, chosen_loss, chosen_acc


def print_cm(cm):
    print("Matriz de confusao - linhas=real, colunas=predito:")
    print("       " + "  ".join([f"{c[:10]:>10s}" for c in CLASSES]))
    for i, cls in enumerate(CLASSES):
        print(f"{cls[:10]:>10s} " + "  ".join([f"{v:10d}" for v in cm[i]]))
    print()


def main():
    print("=== TREINO NAPLES_BLACK v0.1 / NPI ===")
    print(f"Script : {SCRIPT_VERSION}")
    print(f"Root   : {ROOT_DIR}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Saida  : {OUTPUT_DIR}")
    print(f"Modelo : {MODEL_PATH}")
    print(f"Classes: {CLASSES}")
    print()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_DIR}")

    clean_previous_outputs()

    counts = count_images_by_class(DATASET_DIR)
    print("Contagem por classe:")
    for cls in CLASSES:
        print(f"  {cls:16s}: {counts[cls]}")
    print()

    with open(CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script_version": SCRIPT_VERSION,
                "class_names": CLASSES,
                "dataset": str(DATASET_DIR),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_ds, val_ds, train_paths, train_labels, val_paths, val_labels = build_datasets()

    model, base_model = build_model()

    print("\n--- Etapa 1: treino do classificador/topo ---")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=make_callbacks("head", BEST_HEAD_MODEL_PATH),
        verbose=1,
    )

    histories = [("head", history_head)]

    if FINE_TUNE:
        print("\n--- Etapa 2: fine-tuning leve da MobileNetV2 ---")
        base_model.trainable = True

        # Congela quase toda a base.
        for layer in base_model.layers:
            layer.trainable = False

        # Libera apenas as ultimas camadas, mantendo BatchNorm congelado.
        for layer in base_model.layers[-FINE_TUNE_LAST_N_LAYERS:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINE,
            callbacks=make_callbacks("fine", BEST_FINE_MODEL_PATH),
            verbose=1,
        )
        histories.append(("fine", history_fine))

    append_history_csv(histories)

    print("\nSelecionando melhor checkpoint...")
    best_model, chosen_name, eval_loss, eval_acc = choose_best_model(model, val_ds)

    print("Avaliando validacao com o melhor modelo escolhido...")
    eval_loss, eval_acc = best_model.evaluate(val_ds, verbose=1)

    cm = compute_confusion_matrix(best_model, val_ds)
    save_confusion_matrix(cm)
    save_metrics_txt(counts, eval_loss, eval_acc, cm, chosen_name, train_labels, val_labels)

    print("\n=== TREINO CONCLUIDO ===")
    print(f"Modelo final       : {MODEL_PATH}")
    print(f"Melhor checkpoint  : {BEST_MODEL_PATH}")
    print(f"Best head          : {BEST_HEAD_MODEL_PATH}")
    print(f"Best fine          : {BEST_FINE_MODEL_PATH}")
    print(f"Historico CSV      : {HISTORY_CSV}")
    print(f"Split CSV          : {SPLIT_CSV}")
    print(f"Matriz de confusao : {CONFUSION_CSV}")
    print(f"Metricas TXT       : {METRICS_TXT}")
    print(f"Classes JSON       : {CLASS_NAMES_JSON}")
    print()

    print_cm(cm)

    accuracy, macro_precision, macro_recall, macro_f1, per_class = classification_metrics_from_cm(cm)
    print(f"Accuracy CM      : {accuracy:.6f}")
    print(f"Macro precision  : {macro_precision:.6f}")
    print(f"Macro recall     : {macro_recall:.6f}")
    print(f"Macro F1         : {macro_f1:.6f}")
    print()
    print("Metricas por classe:")
    for cls, m in per_class.items():
        print(
            f"  {cls:16s}: precision={m['precision']:.6f} | "
            f"recall={m['recall']:.6f} | f1={m['f1']:.6f} | support={m['support']}"
        )
    print()
    print("Observacao: use como modelo NPI/v0.1 e reforce NG_DESALINHADO quando houver mais amostras reais.")


if __name__ == "__main__":
    main()
