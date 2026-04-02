import os
import random
import shutil
from pathlib import Path

# caminhos
dataset_raw = r"C:\SVC_INSPECAO_USB\dataset_usb"
dataset_out = r"C:\SVC_INSPECAO_USB\dataset"

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

classes = os.listdir(dataset_raw)

for cls in classes:

    src = os.path.join(dataset_raw, cls)
    images = os.listdir(src)

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = images[:n_train]
    val_files = images[n_train:n_train+n_val]
    test_files = images[n_train+n_val:]

    for split, files in zip(
        ["train", "val", "test"],
        [train_files, val_files, test_files]
    ):

        dst_dir = os.path.join(dataset_out, split, cls)
        Path(dst_dir).mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(
                os.path.join(src, f),
                os.path.join(dst_dir, f)
            )

print("\nDataset dividido com sucesso!")