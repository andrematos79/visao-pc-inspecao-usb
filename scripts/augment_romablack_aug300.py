from pathlib import Path
import random
import shutil
import cv2
import numpy as np

SOURCE_DIR = Path("datasets/ROMA_BLACK")
OUTPUT_DIR = Path("datasets/ROMA_BLACK_AUG300")

CLASSES = ["OK", "NG_DESALINHADO", "NG_DANIFICADO"]
TARGET_PER_CLASS = 300
IMG_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(42)
np.random.seed(42)


def list_images(folder: Path):
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def apply_gamma(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


def augment_image(img):
    h, w = img.shape[:2]

    # Rotação, escala e deslocamento leves
    angle = random.uniform(-3.0, 3.0)
    scale = random.uniform(0.97, 1.03)
    tx = random.uniform(-0.03, 0.03) * w
    ty = random.uniform(-0.03, 0.03) * h

    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty

    out = cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Brilho e contraste leves
    alpha = random.uniform(0.90, 1.12)
    beta = random.uniform(-18, 18)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Gamma leve
    gamma = random.uniform(0.90, 1.10)
    out = apply_gamma(out, gamma)

    # Ruído leve
    if random.random() < 0.35:
        sigma = random.uniform(2, 7)
        noise = np.random.normal(0, sigma, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Blur muito leve
    if random.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    # Nitidez leve
    if random.random() < 0.25:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        out = cv2.filter2D(out, -1, kernel)

    return out


def copy_real_images(src_images, out_class_dir):
    out_class_dir.mkdir(parents=True, exist_ok=True)

    for src in src_images:
        dst = out_class_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def main():
    print("=== AUGMENTATION ROMA_BLACK_AUG300 ===")
    print(f"Origem : {SOURCE_DIR}")
    print(f"Destino: {OUTPUT_DIR}")
    print(f"Meta   : {TARGET_PER_CLASS} imagens por classe")
    print()

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Pasta de origem nao encontrada: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        src_class_dir = SOURCE_DIR / class_name
        out_class_dir = OUTPUT_DIR / class_name

        if not src_class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {src_class_dir}")

        real_images = list_images(src_class_dir)

        if not real_images:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {src_class_dir}")

        copy_real_images(real_images, out_class_dir)

        current_count = len(list_images(out_class_dir))

        print(f"{class_name}: atual = {current_count}")

        while current_count < TARGET_PER_CLASS:
            src = random.choice(real_images)
            img = cv2.imread(str(src))

            if img is None:
                print(f"Aviso: nao foi possivel ler {src}")
                continue

            aug = augment_image(img)

            short_stem = src.stem[:40]
            out_name = f"AUG_{class_name}_{current_count + 1:04d}_from_{short_stem}.jpg"
            out_path = out_class_dir / out_name

            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])

            current_count += 1

        print(f"{class_name}: final = {current_count}")
        print()

    print("Concluido.")
    print("Confira visualmente as imagens aumentadas antes de treinar.")


if __name__ == "__main__":
    main()
