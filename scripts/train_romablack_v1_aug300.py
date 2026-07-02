from pathlib import Path
import numpy as np
import tensorflow as tf

DATASET_DIR = Path("datasets/ROMA_BLACK_AUG300")
BASE_MODEL_PATH = Path("models/model_final.keras")
OUTPUT_DIR = Path("models/ROMA_BLACK")

MODEL_OUT = OUTPUT_DIR / "roma_black_v01.keras"
HISTORY_CSV = OUTPUT_DIR / "history_roma_black_v01.csv"
CM_CSV = OUTPUT_DIR / "confusion_matrix_roma_black_v01.csv"
METRICS_TXT = OUTPUT_DIR / "metrics_roma_black_v01.txt"

CLASSES = ["OK", "NG_DESALINHADO", "NG_DANIFICADO"]

BATCH_SIZE = 16
VAL_SPLIT = 0.20
SEED = 42
EPOCHS_HEAD = 6
EPOCHS_FINE = 14

tf.keras.utils.set_random_seed(SEED)


def enable_legacy_batchnorm_compat():
    """
    Compatibilidade para modelos .keras antigos que salvaram BatchNormalization
    com argumentos renorm/renorm_clipping/renorm_momentum.
    O modelo carregado continua usando BatchNormalization oficial do Keras atual.
    """
    BN = tf.keras.layers.BatchNormalization
    original_from_config = BN.from_config

    @classmethod
    def from_config_compat(cls, config):
        config = dict(config)

        for k in [
            "renorm",
            "renorm_clipping",
            "renorm_momentum",
            "fused",
            "virtual_batch_size",
            "adjustment"
        ]:
            config.pop(k, None)

        return original_from_config(config)

    BN.from_config = from_config_compat


enable_legacy_batchnorm_compat()


def load_model_compat(path):
    print(f"Carregando modelo com compatibilidade Keras legado: {path}")

    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path, compile=False)


def detect_preprocess_mode():
    core_path = Path("svc_core_usb_external.py")
    if not core_path.exists():
        return "mobilenetv2"

    txt = core_path.read_text(encoding="utf-8", errors="ignore").lower()

    if "preprocess_input" in txt:
        return "mobilenetv2"

    if "/255.0" in txt or "/ 255.0" in txt or "/255" in txt or "/ 255" in txt:
        return "rescale01"

    return "mobilenetv2"


def preprocess_dataset(ds, mode):
    if mode == "rescale01":
        return ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, tf.float32)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )


def get_model_input_size(model):
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    h = shape[1] if len(shape) > 1 and shape[1] is not None else 224
    w = shape[2] if len(shape) > 2 and shape[2] is not None else 224

    return int(h), int(w)


def count_trainable_params(model):
    total = 0
    for w in model.trainable_weights:
        total += int(np.prod(w.shape))
    return total


def save_confusion_and_metrics(cm):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savetxt(CM_CSV, cm, fmt="%d", delimiter=",", header=",".join(CLASSES), comments="")

    lines = []
    lines.append("=== ROMA_BLACK v01 AUG300 - VALIDATION METRICS ===")
    lines.append("")
    lines.append("Classes: " + ", ".join(CLASSES))
    lines.append("")
    lines.append("Confusion matrix rows=true cols=pred:")
    lines.append(str(cm))
    lines.append("")

    total_correct = np.trace(cm)
    total = np.sum(cm)
    acc = total_correct / total if total > 0 else 0.0

    lines.append(f"Accuracy: {acc:.4f}")
    lines.append("")

    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        pred_total = cm[:, i].sum()
        true_total = cm[i, :].sum()

        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / true_total if true_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        lines.append(f"{cls}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} support={true_total}")

    METRICS_TXT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"Confusion matrix salva em: {CM_CSV}")
    print(f"Metricas salvas em: {METRICS_TXT}")


def main():
    print("=== TREINO ROMA_BLACK v01 AUG300 ===")
    print(f"Dataset     : {DATASET_DIR}")
    print(f"Base model  : {BASE_MODEL_PATH}")
    print(f"Saida modelo: {MODEL_OUT}")
    print()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_DIR}")

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo base nao encontrado: {BASE_MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando modelo base...")
    model = load_model_compat(BASE_MODEL_PATH)

    if int(model.output_shape[-1]) != len(CLASSES):
        raise RuntimeError(
            f"Modelo base tem {model.output_shape[-1]} saidas, mas o ROMA_BLACK precisa de {len(CLASSES)} classes."
        )

    img_h, img_w = get_model_input_size(model)
    print(f"Input size detectado: {img_w}x{img_h}")

    preprocess_mode = "raw_0_255_model_has_rescaling"
    print(f"Preprocess forçado: {preprocess_mode}")
    print()

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(img_h, img_w),
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="training"
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(img_h, img_w),
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="validation"
    )

    # O modelo base já possui camada interna de Rescaling.
    # Portanto, não aplicamos preprocess_input aqui.
    train_ds = train_ds_raw.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_raw.prefetch(tf.data.AUTOTUNE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(HISTORY_CSV))
    ]

    print("Fase 1: treinando cabeca/classificador...")
    for layer in model.layers:
        layer.trainable = False

    model.layers[-1].trainable = True

    if count_trainable_params(model) == 0:
        for layer in model.layers[-3:]:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks
    )

    print()
    print("Fase 2: fine-tuning leve do modelo completo...")

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=EPOCHS_HEAD,
        epochs=EPOCHS_HEAD + EPOCHS_FINE,
        callbacks=callbacks
    )

    print()
    print("Carregando melhor modelo salvo para avaliacao final...")
    best_model = tf.keras.models.load_model(MODEL_OUT, compile=False)

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    for images, labels in val_ds:
        preds = best_model.predict(images, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = labels.numpy().astype(int)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

    save_confusion_and_metrics(cm)

    print()
    print("Treino concluido.")
    print(f"Modelo final/best salvo em: {MODEL_OUT}")


if __name__ == "__main__":
    main()


