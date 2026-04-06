import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIGURAÇÕES
# =========================
TEST_DIR = Path(r"C:\SVC_INSPECAO_USB\dataset_usb_v1\test")
MODEL_PATH = Path(r"C:\SVC_INSPECAO_USB\models\usb_mobilenetv2\best_model.keras")
OUTPUT_DIR = Path(r"C:\SVC_INSPECAO_USB\reports\usb_model_evaluation")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "misclassified").mkdir(parents=True, exist_ok=True)

# =========================
# DATASET DE TESTE
# =========================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("\nClasses detectadas:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# Guardar caminhos reais dos arquivos na mesma ordem do dataset
file_paths = []
for class_name in class_names:
    class_dir = TEST_DIR / class_name
    class_files = sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])
    file_paths.extend(class_files)

# =========================
# MODELO
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# INFERÊNCIA
# =========================
y_true = []
y_pred = []
y_prob = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    y_prob.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# Segurança
if len(file_paths) != len(y_true):
    print(f"\n[AVISO] Quantidade de arquivos ({len(file_paths)}) diferente do total de amostras ({len(y_true)}).")
    print("A análise ainda será feita, mas a cópia de erros pode ficar desalinhada.")
else:
    print(f"\nTotal de arquivos alinhados com o dataset: {len(file_paths)}")

# =========================
# MATRIZ DE CONFUSÃO
# =========================
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_csv_path = OUTPUT_DIR / "confusion_matrix.csv"
cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de Confusão - SVC USB")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

threshold = cm.max() / 2.0 if cm.max() > 0 else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("Classe Real")
plt.xlabel("Classe Predita")
plt.tight_layout()
cm_png_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_png_path, dpi=200, bbox_inches="tight")
plt.close()

# =========================
# RELATÓRIO DE CLASSIFICAÇÃO
# =========================
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_txt = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = OUTPUT_DIR / "classification_report.csv"
report_txt_path = OUTPUT_DIR / "classification_report.txt"

report_df.to_csv(report_csv_path, encoding="utf-8-sig")
with open(report_txt_path, "w", encoding="utf-8") as f:
    f.write(report_txt)

print("\n=== CLASSIFICATION REPORT ===")
print(report_txt)

# =========================
# TABELA DE PREVISÕES
# =========================
rows = []
for idx in range(len(y_true)):
    true_idx = int(y_true[idx])
    pred_idx = int(y_pred[idx])
    probs = y_prob[idx]

    row = {
        "arquivo": str(file_paths[idx]) if idx < len(file_paths) else f"sample_{idx}",
        "classe_real": class_names[true_idx],
        "classe_predita": class_names[pred_idx],
        "acertou": true_idx == pred_idx,
        "prob_predita": float(np.max(probs)),
    }

    for c_idx, c_name in enumerate(class_names):
        row[f"prob_{c_name}"] = float(probs[c_idx])

    rows.append(row)

pred_df = pd.DataFrame(rows)
pred_csv_path = OUTPUT_DIR / "predictions.csv"
pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

# =========================
# COPIAR ERROS
# =========================
misclassified_dir = OUTPUT_DIR / "misclassified"

# limpar pasta anterior
for item in misclassified_dir.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

error_count = 0
for idx, row in pred_df.iterrows():
    if not row["acertou"]:
        error_count += 1
        true_class = row["classe_real"]
        pred_class = row["classe_predita"]

        dst_dir = misclassified_dir / true_class / f"pred_{pred_class}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_path = Path(row["arquivo"])
        if src_path.exists():
            dst_name = f"{src_path.stem}__REAL_{true_class}__PRED_{pred_class}{src_path.suffix}"
            shutil.copy2(src_path, dst_dir / dst_name)

# =========================
# RESUMO
# =========================
accuracy = (y_true == y_pred).mean()

summary_path = OUTPUT_DIR / "summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("SVC USB - Avaliação do Modelo\n")
    f.write("=" * 40 + "\n")
    f.write(f"Modelo: {MODEL_PATH}\n")
    f.write(f"Diretório de teste: {TEST_DIR}\n")
    f.write(f"Total de amostras: {len(y_true)}\n")
    f.write(f"Acurácia geral: {accuracy:.4f}\n")
    f.write(f"Total de erros: {error_count}\n\n")
    f.write("Classes:\n")
    for i, c in enumerate(class_names):
        f.write(f"{i}: {c}\n")

print("\n=== RESUMO ===")
print(f"Acurácia geral : {accuracy:.4f}")
print(f"Total de erros : {error_count}")
print(f"Matriz CSV     : {cm_csv_path}")
print(f"Matriz PNG     : {cm_png_path}")
print(f"Report CSV     : {report_csv_path}")
print(f"Report TXT     : {report_txt_path}")
print(f"Predictions    : {pred_csv_path}")
print(f"Erros salvos   : {misclassified_dir}")
print(f"Summary        : {summary_path}")

# =========================
# SALVAR LABELS
# =========================
labels_path = OUTPUT_DIR / "labels_detected.json"
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2, ensure_ascii=False)

print(f"Labels         : {labels_path}")