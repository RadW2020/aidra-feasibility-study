#!/bin/bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "=== AIDRA Model Download ==="
echo "Target directory: $MODELS_DIR"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is required but not found."
    exit 1
fi

# Check if ultralytics is installed
if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "ERROR: ultralytics package is required. Install with: pip install ultralytics"
    exit 1
fi

# Download YOLOv8 nano (base model)
if [ ! -f "$MODELS_DIR/yolov8n.pt" ]; then
    echo "Downloading YOLOv8n base model (~6 MB)..."
    python3 -c "
from ultralytics import YOLO
import shutil
model = YOLO('yolov8n.pt')
shutil.move('yolov8n.pt', '$MODELS_DIR/yolov8n.pt')
print('  Downloaded successfully.')
"
else
    echo "  YOLOv8n already exists, skipping."
fi

# Download YOLOv8 small (for comparison)
if [ ! -f "$MODELS_DIR/yolov8s.pt" ]; then
    echo "Downloading YOLOv8s model (~22 MB)..."
    python3 -c "
from ultralytics import YOLO
import shutil
model = YOLO('yolov8s.pt')
shutil.move('yolov8s.pt', '$MODELS_DIR/yolov8s.pt')
print('  Downloaded successfully.')
"
else
    echo "  YOLOv8s already exists, skipping."
fi

# Download SAR ship detection YOLOv8m (from HuggingFace, Apache 2.0)
if [ ! -f "$MODELS_DIR/vesseltracker-sar-yolov8.pt" ]; then
    echo "Downloading vesselTracker SAR YOLOv8m (~50 MB)..."
    curl -L -o "$MODELS_DIR/vesseltracker-sar-yolov8.pt" \
        "https://huggingface.co/hewitleo/sar-ship-detection-yolov8/resolve/main/weights_(model)/best.pt"
    echo "  Downloaded successfully."
else
    echo "  vesseltracker-sar-yolov8.pt already exists, skipping."
fi

# Export YOLOv8n to ONNX
if [ -f "$MODELS_DIR/yolov8n.pt" ] && [ ! -f "$MODELS_DIR/yolov8n.onnx" ]; then
    echo "Exporting YOLOv8n to ONNX..."
    python3 -c "
from ultralytics import YOLO
import shutil
model = YOLO('$MODELS_DIR/yolov8n.pt')
path = model.export(format='onnx', opset=13)
shutil.move(path, '$MODELS_DIR/yolov8n.onnx')
print('  Exported successfully.')
"
else
    echo "  YOLOv8n ONNX already exists, skipping."
fi

echo ""
echo "=== Model Checksums ==="
for f in "$MODELS_DIR"/*.pt "$MODELS_DIR"/*.onnx 2>/dev/null; do
    if [ -f "$f" ]; then
        hash=$(shasum -a 256 "$f" | cut -d' ' -f1)
        size=$(du -h "$f" | cut -f1)
        echo "  $size  $hash  $(basename "$f")"
    fi
done

echo ""
echo "=== Done ==="
ls -lh "$MODELS_DIR/"
