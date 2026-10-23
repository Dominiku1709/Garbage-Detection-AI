# Garbage Detection AI

A local Streamlit dashboard for waste classification, detection history, and recycling guidance. The project includes YOLO/RT-DETR research code and this dashboard uses a configured local YOLO checkpoint when available. No paid APIs are used.

## Run the dashboard

Use Python 3.10 or 3.11 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run main.py
```

The app is responsive and offers an overview dashboard, image analysis, persistent history, a recycling guide, and the existing live-camera workflow. Uploaded images are analyzed in memory and are not saved; filenames, results, model labels, and timestamps are stored in `output/waste_history.sqlite3`.

## Configure an AI checkpoint

The repository does not include trained weights: the project's `FINAL` model directory is intentionally excluded from version control. Set `GARBAGE_MODEL_PATH` to a local Ultralytics-compatible YOLO checkpoint trained for waste classes. In PowerShell:

```powershell
$env:GARBAGE_MODEL_PATH = "C:\models\waste-best.pt"
streamlit run main.py
```

The app also checks the two YOLO checkpoint locations used by the existing project under `FINAL/YOLO_DINOV3` and `FINAL/YOLO_CODER`. If a checkpoint is configured but cannot be loaded, the app displays the error rather than silently switching classifiers. Detection labels are mapped into plastic, paper, metal, glass, organic waste, and other waste. Model confidence is shown for each detection.

Without a local checkpoint, the app remains usable in **manual-review fallback** mode: upload an image, choose its category after reviewing it, and get the corresponding sorting recommendation. This mode is explicitly not AI detection and does not invent a confidence score. To obtain model predictions, provide a locally trained checkpoint; no external API or model download is required.

## History and data

The dashboard stores image filenames, category counts, confidence values, model name, and local timestamp in SQLite under `output/`. It does not retain uploaded image bytes. Removing `output/waste_history.sqlite3` clears the dashboard history.

## Existing model research code

The `rtdetr/`, `models/`, and `inference/` directories contain the project's existing RT-DETR and YOLO training/inference components. RT-DETR checkpoints and generated configs are not bundled; see `rtdetr/README.md` and `rtdetr/configs/rtdetrv2/rtdetrv2_taco_finetune_BASELINE.yml.template` for the model training setup.
