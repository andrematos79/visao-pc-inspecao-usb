from pathlib import Path
import random
import shutil

import cv2
import numpy as np

# ============================================================
# AUGMENTATION CONTROLADO - SVC USB / NAPLES_BLACK_AUG200
#
# Objetivo:
#   Usar o dataset real NAPLES_BLACK como origem e montar o
#   dataset NAPLES_BLACK_AUG200 com 200 imagens por classe.
#
# Estrutura esperada:
#   C:\SVC_INSPECAO_USB_GIT\datasets\NAPLES_BLACK\OK
#   C:\SVC_INSPECAO_USB_GIT\datasets\NAPLES_BLACK\NG_DESALINHADO
#   C:\SVC_INSPECAO_USB_GIT\datasets\NAPLES_BLACK\NG_DANIFICADO
#
# Saida:
#   C:\SVC_INSPECAO_USB_GIT\datasets\NAPLES_BLACK_AUG200\...
#
# Observacao importante:
#   Nao usa flip horizontal/vertical para evitar criar geometrias
#   irreais do conector USB.
# ============================================================

# Preferencialmente execute este script a partir da raiz do projeto:
#   cd C:\SVC_INSPECAO_USB_GIT
#   python .\augment_naples_black_aug200.py

SOURCE_DIR = Path("datasets/NAPLES_BLACK")
OUTPUT_DIR = Path("datasets/NAPLES_BLACK_AUG200")

CLASSES = ["OK", "NG_DESALINHADO", "NG_DANIFICADO"]
TARGET_PER_CLASS = 200
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Seed fixa para permitir reproduzir aproximadamente o mesmo dataset aumentado.
random.seed(42)
np.random.seed(42)


def resolve_project_relative(path: Path) -> Path:
    """Resolve o caminho em relacao ao diretorio atual ou ao local do script."""
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists() or (Path.cwd() / "datasets").exists():
        return cwd_path

    script_dir = Path(__file__).resolve().parent
    return script_dir / path


def list_images(folder: Path):
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def apply_gamma(img, gamma: float):
    gamma = max(0.10, float(gamma))
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


def augment_image(img):
    """
    Aplica augmentation leve/controlado.

    Transformacoes usadas:
      - pequena rotacao
      - pequeno zoom
      - pequeno deslocamento
      - ajuste leve de brilho/contraste/gamma
      - ruido leve opcional
      - blur ou nitidez leve opcional

    Nao usa flip para nao inverter caracteristicas reais do conector.
    """
    h, w = img.shape[:2]

    # Rotacao, escala e deslocamento leves
    angle = random.uniform(-2.5, 2.5)
    scale = random.uniform(0.98, 1.02)
    tx = random.uniform(-0.02, 0.02) * w
    ty = random.uniform(-0.02, 0.02) * h

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    out = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Brilho e contraste leves
    alpha = random.uniform(0.92, 1.08)
    beta = random.uniform(-12, 12)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Gamma leve
    gamma = random.uniform(0.93, 1.07)
    out = apply_gamma(out, gamma)

    # Ruido leve
    if random.random() < 0.30:
        sigma = random.uniform(1.5, 5.0)
        noise = np.random.normal(0, sigma, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Blur muito leve
    if random.random() < 0.20:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    # Nitidez leve
    if random.random() < 0.20:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ])
        out = cv2.filter2D(out, -1, kernel)

    return out


def copy_real_images(src_images, out_class_dir: Path):
    """Copia imagens reais para o dataset aumentado sem sobrescrever."""
    out_class_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in src_images:
        dst = out_class_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    return copied


def unique_output_path(folder: Path, filename: str) -> Path:
    """Garante que o nome de saida nao sobrescreva arquivo existente."""
    candidate = folder / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    idx = 1
    while True:
        candidate = folder / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def generate_until_target(class_name: str, real_images, out_class_dir: Path):
    current_count = len(list_images(out_class_dir))
    generated = 0

    while current_count < TARGET_PER_CLASS:
        src = random.choice(real_images)
        img = cv2.imread(str(src))

        if img is None:
            print(f"  Aviso: nao foi possivel ler {src}")
            continue

        aug = augment_image(img)

        short_stem = src.stem[:45]
        out_name = f"AUG_{class_name}_{current_count + 1:04d}_from_{short_stem}.jpg"
        out_path = unique_output_path(out_class_dir, out_name)

        ok = cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            print(f"  Aviso: nao foi possivel salvar {out_path}")
            continue

        current_count += 1
        generated += 1

    return generated, current_count


def main():
    source_dir = resolve_project_relative(SOURCE_DIR)
    output_dir = resolve_project_relative(OUTPUT_DIR)

    print("=== AUGMENTATION NAPLES_BLACK_AUG200 ===")
    print(f"Origem : {source_dir}")
    print(f"Destino: {output_dir}")
    print(f"Meta   : {TARGET_PER_CLASS} imagens por classe")
    print()

    if not source_dir.exists():
        raise FileNotFoundError(f"Pasta de origem nao encontrada: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_final = 0

    for class_name in CLASSES:
        src_class_dir = source_dir / class_name
        out_class_dir = output_dir / class_name

        if not src_class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {src_class_dir}")

        real_images = list_images(src_class_dir)
        if not real_images:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {src_class_dir}")

        out_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Classe: {class_name}")
        print(f"  Reais na origem       : {len(real_images)}")

        if len(real_images) < 50:
            print("  Atencao: classe com poucas imagens reais. Use este modelo como NPI/v0.1.")

        copied = copy_real_images(real_images, out_class_dir)
        current_after_copy = len(list_images(out_class_dir))

        print(f"  Reais copiadas agora  : {copied}")
        print(f"  Atual apos copia      : {current_after_copy}")

        if current_after_copy >= TARGET_PER_CLASS:
            print(f"  Final                 : {current_after_copy} (ja atingiu a meta)")
            print()
            total_final += current_after_copy
            continue

        generated, final_count = generate_until_target(class_name, real_images, out_class_dir)

        print(f"  Augmentations geradas : {generated}")
        print(f"  Final                 : {final_count}")
        print()

        total_final += final_count

    print("Concluido.")
    print(f"Total final esperado: {TARGET_PER_CLASS * len(CLASSES)} imagens")
    print(f"Total final contado : {total_final} imagens")
    print("Confira visualmente algumas imagens aumentadas antes de treinar.")


if __name__ == "__main__":
    main()
