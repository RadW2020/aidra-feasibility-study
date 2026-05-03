"""
Fine-tuning de YOLOv8 con dataset xView3-SAR para vessel detection.

Este script se ejecuta en una maquina con GPU (no en OCI ARM).
El modelo fine-tuned se copia luego a models/ para inferencia en ARM.

Prerequisitos:
- GPU con >= 8 GB VRAM (o Google Colab)
- Dataset xView3-SAR descargado y convertido a formato YOLO
- pip install ultralytics torch

Uso:
    python scripts/fine-tune.py --data /path/to/xview3-sar/data.yaml
    python scripts/fine-tune.py --data /path/to/data.yaml --epochs 100 --model yolov8s.pt
    python scripts/fine-tune.py --data /path/to/data.yaml --export-only --weights runs/best.pt

Dataset xView3-SAR:
    Descargar de https://iuu.xview.us/
    243,018 objetos maritimos anotados en imagenes Sentinel-1
    Convertir a formato YOLO: images/ + labels/ con data.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path


def create_data_yaml(dataset_dir: Path, output_path: Path) -> Path:
    """Create a YOLO data.yaml file for the dataset."""
    content = f"""
# xView3-SAR Dataset for AIDRA Vessel Detection
path: {dataset_dir}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: vessel
"""
    output_path.write_text(content.strip())
    print(f"  Created data.yaml at {output_path}")
    return output_path


def fine_tune(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/aidra",
    name: str = "vessel-detection",
) -> Path:
    """
    Fine-tune YOLOv8 on vessel detection dataset.

    Returns path to best weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("=== AIDRA Fine-Tuning ===")
    print(f"  Base model:  {base_model}")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Image size:  {imgsz}")
    print(f"  Batch size:  {batch}")
    print(f"  Project:     {project}/{name}")
    print()

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Augmentation
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        # Hardware
        device="0",  # First GPU
        workers=8,
        # Logging
        verbose=True,
        plots=True,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(project) / name / "weights" / "last.pt"

    print("\n=== Training Complete ===")
    print(f"  Best weights: {best_weights}")

    return best_weights


def export_model(weights_path: str, output_dir: str = "models") -> None:
    """Export trained model to multiple formats."""
    from ultralytics import YOLO

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=== Exporting Model ===")
    print(f"  Weights: {weights_path}")

    model = YOLO(weights_path)

    # Copy PyTorch weights
    pt_dest = output / "yolov8n-sar.pt"
    shutil.copy2(weights_path, pt_dest)
    print(f"  PyTorch:  {pt_dest} ({pt_dest.stat().st_size / 1e6:.1f} MB)")

    # Export to ONNX
    onnx_path = model.export(format="onnx", opset=13, simplify=True)
    onnx_dest = output / "yolov8n-sar.onnx"
    shutil.move(onnx_path, onnx_dest)
    print(f"  ONNX:     {onnx_dest} ({onnx_dest.stat().st_size / 1e6:.1f} MB)")

    print("\n=== Export Complete ===")
    print(f"  Models saved to {output}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for AIDRA vessel detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml (YOLO format dataset config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for exported models (default: models/)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training, only export from --weights",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights (for --export-only)",
    )

    args = parser.parse_args()

    if args.export_only:
        if not args.weights:
            parser.error("--export-only requires --weights")
        export_model(args.weights, args.output)
    else:
        best_weights = fine_tune(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        export_model(str(best_weights), args.output)


if __name__ == "__main__":
    main()
