import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# CONFIG
# =========================================================
DATASET_DIR = r"C:\SVC_INSPECAO_USB\dataset"
OUTPUT_DIR = r"C:\SVC_INSPECAO_USB\outputs_usb_mobilenetv2_finetune"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# fase 1
EPOCHS_HEAD = 15
LR_HEAD = 1e-4

# fase 2 (fine-tuning)
EPOCHS_FINE = 15
LR_FINE = 1e-5
UNFREEZE_LAST_LAYERS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

# =========================================================
# DATASETS
# =========================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses encontradas:", class_names)

labels_path = os.path.join(OUTPUT_DIR, "labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# =========================================================
# DATA AUGMENTATION
# =========================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomContrast(0.08),
], name="data_augmentation")

# =========================================================
# MODEL
# =========================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.30)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

# =========================================================
# CALLBACKS
# =========================================================
best_head_path = os.path.join(OUTPUT_DIR, "best_head.keras")
best_finetuned_path = os.path.join(OUTPUT_DIR, "best_finetuned.keras")
final_model_path = os.path.join(OUTPUT_DIR, "model_final.keras")

callbacks_head = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        best_head_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

callbacks_fine = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        best_finetuned_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# =========================================================
# FASE 1 - TREINAR CABEÇA
# =========================================================
print("\n==============================")
print("FASE 1 - Treino da cabeça")
print("==============================")

model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head
)

# =========================================================
# FASE 2 - FINE-TUNING
# =========================================================
print("\n==============================")
print("FASE 2 - Fine-tuning")
print("==============================")

# carrega melhor modelo da fase 1
model = load_model(best_head_path)

# localizar base_model dentro do modelo carregado
base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise RuntimeError("Não foi possível localizar a base MobileNetV2 dentro do modelo.")

base_model.trainable = True

# congela tudo, exceto últimas camadas
for layer in base_model.layers[:-UNFREEZE_LAST_LAYERS]:
    layer.trainable = False

for layer in base_model.layers[-UNFREEZE_LAST_LAYERS:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine
)

# carregar melhor modelo final
if os.path.exists(best_finetuned_path):
    model = load_model(best_finetuned_path)

model.save(final_model_path)

# =========================================================
# HISTÓRICO COMBINADO
# =========================================================
acc = history_head.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history_head.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history_head.history["loss"] + history_fine.history["loss"]
val_loss = history_head.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="train_accuracy")
plt.plot(val_acc, label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

# =========================================================
# TESTE FINAL
# =========================================================
print("\n==============================")
print("AVALIAÇÃO FINAL NO TESTE")
print("==============================")

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

print("\nClassification Report:\n")
print(report)

# salvar report txt
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

# salvar confusion matrix txt
cm_txt_path = os.path.join(OUTPUT_DIR, "confusion_matrix.txt")
with open(cm_txt_path, "w", encoding="utf-8") as f:
    f.write("Classes:\n")
    f.write(str(class_names) + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

# salvar confusion matrix png
fig, ax = plt.subplots(figsize=(7, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix - SVC USB")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# =========================================================
# RESULTS JSON
# =========================================================
results = {
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_head": EPOCHS_HEAD,
    "epochs_fine": EPOCHS_FINE,
    "learning_rate_head": LR_HEAD,
    "learning_rate_fine": LR_FINE,
    "unfreeze_last_layers": UNFREEZE_LAST_LAYERS,
    "classes": class_names,
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss)
}

results_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nTreinamento finalizado. Arquivos salvos em: {OUTPUT_DIR}")