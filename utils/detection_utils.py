"""
detection_utils.py
------------------
Object-detection lesson helpers — **results only**.

The sibling ``object-detection/`` repo runs YOLO (Ultralytics) and exports its
outputs to ``object-detection/results/`` with ``export_results.py``: every box
both models found on a set of openly licensed photos, plus the fine-tuning
metrics. This module only *reads* those files. No torch, no YOLO in the app —
that keeps the app light (torch is ~2 GB), and it keeps the app out of the
AGPL licence that Ultralytics YOLO and models trained with it carry.

Because every box is stored down to a low confidence, the page can still be
interactive: moving the confidence slider redraws the boxes here with Pillow,
exactly as the model would have drawn them at that threshold.

Streamlit-free: returns data and images, renders nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "object-detection"
RESULTS = SIBLING / "results"

MODEL_NAMES = {
    "pretrained": "Pretrained YOLO (80 everyday things)",
    "fine_tuned": "Fine-tuned YOLO (spaghetti bolognese)",
}

# A fixed, colour-blind-friendly cycle so one class keeps one colour.
BOX_COLOURS = ["#2E86DE", "#E67E22", "#27AE60", "#C0392B", "#8E44AD", "#16A085", "#D35400", "#2C3E50"]


# ---------------------------------------------------------------------------
# Loading what the sibling exported
# ---------------------------------------------------------------------------
def load_results(results_dir: Path = RESULTS) -> dict | None:
    """The exported ``detections.json``, or ``None`` when the sibling isn't there."""
    path = Path(results_dir) / "detections.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def detections_frame(results: dict, model: str) -> pd.DataFrame:
    """Every box one model found, one row per box (pixels, top-left origin)."""
    rows = []
    for image, entry in results["models"][model]["images"].items():
        for det in entry["detections"]:
            x1, y1, x2, y2 = det["box"]
            rows.append({
                "image": image, "label": det["label"], "confidence": det["confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    columns = ["image", "label", "confidence", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=columns)


def images_for(results: dict, model: str, folder: str | None = None) -> list[str]:
    """Image keys (``folder/file.jpg``) a model was run on, optionally one folder."""
    keys = sorted(results["models"][model]["images"])
    return [k for k in keys if folder is None or k.startswith(f"{folder}/")]


def image_path(key: str, results_dir: Path = RESULTS) -> Path:
    """The original (box-free) photo for an image key."""
    return Path(results_dir) / "images" / key


# ---------------------------------------------------------------------------
# The confidence threshold — the one knob a user of a detector really turns
# ---------------------------------------------------------------------------
def above(frame: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Keep the boxes the model is at least this sure about."""
    return frame[frame["confidence"] >= min_confidence].reset_index(drop=True)


def threshold_table(frame: pd.DataFrame, thresholds=(0.1, 0.25, 0.5, 0.75)) -> pd.DataFrame:
    """How many boxes survive each threshold — the trade-off in one table.

    Low threshold: you find everything, plus junk. High threshold: only sure
    things, but you miss some. Which is right depends on the cost of a miss.
    """
    return pd.DataFrame({
        "minimum confidence": [f"{t:.0%}" for t in thresholds],
        "boxes kept": [int((frame["confidence"] >= t).sum()) for t in thresholds],
    })


def label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """What was found: count and best confidence per label, most frequent first."""
    if frame.empty:
        return pd.DataFrame(columns=["label", "count", "best confidence"])
    return (
        frame.groupby("label")["confidence"]
        .agg(count="size", best="max")
        .rename(columns={"best": "best confidence"})
        .sort_values(["count", "best confidence"], ascending=False)
        .reset_index()
    )


def colour_for(label: str, labels: list[str]) -> str:
    return BOX_COLOURS[sorted(labels).index(label) % len(BOX_COLOURS)] if label in labels else BOX_COLOURS[0]


def draw_boxes(path: Path, frame: pd.DataFrame, line_width: int = 4) -> Image.Image:
    """The photo with one labelled box per row of ``frame`` — what YOLO draws."""
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 45))
    labels = sorted(frame["label"].unique())
    for row in frame.sort_values("confidence").itertuples():
        colour = colour_for(row.label, labels)
        draw.rectangle([row.x1, row.y1, row.x2, row.y2], outline=colour, width=line_width)
        text = f"{row.label} {row.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((row.x1, row.y1), text, font=font)
        lift = bottom - top + 6
        y = row.y1 - lift if row.y1 - lift >= 0 else row.y1
        draw.rectangle([row.x1, y, row.x1 + (right - left) + 8, y + lift], fill=colour)
        draw.text((row.x1 + 4, y + 2), text, fill="white", font=font)
    return image


# ---------------------------------------------------------------------------
# Labels and the overlap score behind "mAP50"
# ---------------------------------------------------------------------------
def to_yolo_label(box: list[float], width: int, height: int, class_id: int = 0) -> str:
    """A pixel box → the one-line text format YOLO trains on.

    ``class  x_centre  y_centre  box_width  box_height``, all as fractions of the
    image size, so the label stays valid when the photo is resized. This is what
    Label Studio writes when you export a project "in YOLO format".
    """
    x1, y1, x2, y2 = box
    return (
        f"{class_id} {(x1 + x2) / 2 / width:.4f} {(y1 + y2) / 2 / height:.4f} "
        f"{(x2 - x1) / width:.4f} {(y2 - y1) / height:.4f}"
    )


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over union: how much two boxes overlap, from 0 to 1.

    A predicted box counts as a hit when it overlaps the hand-drawn box enough.
    "mAP50" uses 50 % overlap as the bar; "mAP50-95" averages over stricter bars
    up to 95 %, which is why it is always the lower number.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Fine-tuning: metrics, the plain-language reading of them, the photo credits
# ---------------------------------------------------------------------------
def training_metrics(results_dir: Path = RESULTS) -> pd.DataFrame:
    """Per-epoch metrics from the fine-tuning run, with readable column names."""
    frame = pd.read_csv(Path(results_dir) / "training" / "results.csv")
    frame.columns = frame.columns.str.strip()
    return frame.rename(columns={
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
        "train/box_loss": "train box loss",
        "val/box_loss": "validation box loss",
        "train/cls_loss": "train class loss",
        "val/cls_loss": "validation class loss",
    })


def epochs_to_reach(metrics: pd.DataFrame, column: str = "mAP50", level: float = 0.9) -> int | None:
    """First epoch at which a metric reaches a level — "how long did learning take?"."""
    hit = metrics.loc[metrics[column] >= level, "epoch"]
    return int(hit.iloc[0]) if len(hit) else None


def compare_models(results: dict, folder: str = "spaghetti", min_confidence: float = 0.25) -> pd.DataFrame:
    """Per photo: what the pretrained model saw vs. what the fine-tuned one saw.

    The pretrained model has no word for spaghetti bolognese; at best it finds
    a "bowl" or "dining table". After a few dozen labelled photos the fine-tuned model
    has one — and the interesting rows are the look-alikes (gnocchi, pizza).
    """
    rows = []
    for key in images_for(results, "fine_tuned", folder):
        row = {"photo": Path(key).stem.replace("_", " ")}
        for model, column in [("pretrained", "pretrained sees"), ("fine_tuned", "fine-tuned sees")]:
            found = above(detections_frame(results, model).query("image == @key"), min_confidence)
            row[column] = (
                ", ".join(f"{r.label} ({r.confidence:.0%})" for r in found.sort_values("confidence", ascending=False).itertuples())
                or "—"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_credits(results_dir: Path = RESULTS) -> pd.DataFrame:
    """The photo credits table from ``results/CREDITS.md`` (markdown → rows)."""
    path = Path(results_dir) / "CREDITS.md"
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            rows.append(dict(zip(["file", "original", "author", "license"], cells)))
    return pd.DataFrame(rows)


def credits_markdown(results_dir: Path = RESULTS) -> str:
    """One compact attribution line per photo (CC BY requires naming the author)."""
    return "\n".join(
        f"- {row.file.strip('`')}: {row.original} — {row.author}, {row.license}"
        for row in load_credits(results_dir).itertuples()
    )


# ---------------------------------------------------------------------------
# "All the code it took" — the sibling notebook, read-only
# ---------------------------------------------------------------------------
def notebook_cells(path: Path = SIBLING / "showcase.ipynb") -> list[tuple[str, str]]:
    """The notebook as ``(cell_type, source)`` pairs, outputs dropped."""
    notebook = json.loads(Path(path).read_text())
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]


def code_line_count(cells: list[tuple[str, str]]) -> int:
    """Lines of real code (no blanks, no comments) — the "it's not much" number."""
    return sum(
        1
        for kind, source in cells if kind == "code"
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def workflow_dot(training: dict | None = None) -> str:
    """The whole project as one picture: photos → labels → fine-tune → detector."""
    training = training or {}
    photos = training.get("train_images", 0) + training.get("val_images", 0)
    rounds = training.get("epochs", "?")
    minutes = training.get("training_minutes")
    took = f"~{minutes:.0f} min on a laptop CPU" if minutes else ""
    return f"""
digraph G {{
  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#EAF2FB"];
  photos [label="📷 {photos} photos\\nof the new thing"];
  label  [label="🏷️ Label Studio\\ndraw a box per photo"];
  yolo   [label="🧠 YOLO pretrained\\n(knows 80 things)", fillcolor="#FDF2E9"];
  train  [label="🔁 fine-tune\\n{rounds} rounds, {took}"];
  model  [label="🍝 new detector\\nknows spaghetti", fillcolor="#E9F7EF"];
  photos -> label -> train; yolo -> train; train -> model;
}}
"""
