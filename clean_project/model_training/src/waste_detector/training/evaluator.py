"""
Evaluation metric extraction for trained YOLO checkpoints.

Runs Ultralytics validation on one or more dataset splits and collects
precision, recall, F1, mAP50, mAP75, and mAP50-95 per split and per class.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from waste_detector.utils.io import write_json


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_f1(box: Any) -> Optional[float]:
    """
    Compute mean F1 from per-class precision and recall arrays.
    Returns None if the arrays are not available or are empty.
    """
    try:
        p = np.asarray(box.p, dtype=float)   # Per-class precision
        r = np.asarray(box.r, dtype=float)   # Per-class recall
        denom = p + r
        # Avoid division by zero for classes with no predictions.
        f1_per_class = np.where(denom > 0, 2 * p * r / denom, 0.0)
        return float(f1_per_class.mean())
    except Exception:
        return None


def _reduce_confusion_matrix(
    val_results: Any,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Pull the confusion matrix off 'val_results' and return the version with
    its background row/column removed, plus the class names that correspond
    to the remaining rows and columns.

    Ultralytics exposes the confusion matrix as 'val_results.confusion_matrix'
    with shape '(nc + 1, nc + 1)'; the last row/column is the background
    class. This helper returns 'None' when the matrix is not available or
    has an unexpected shape so the caller can degrade gracefully.
    """
    cm_obj = getattr(val_results, "confusion_matrix", None)
    if cm_obj is None or getattr(cm_obj, "matrix", None) is None:
        return None

    # Convert to a plain float array so the arithmetic below is safe.
    full_matrix = np.asarray(cm_obj.matrix, dtype=float)
    if full_matrix.ndim != 2 or full_matrix.shape[0] != full_matrix.shape[1]:
        return None

    # Drop the trailing background row and column.
    nc = full_matrix.shape[0] - 1
    if nc < 1:
        return None
    reduced = full_matrix[:nc, :nc]

    # Look up class names so plots and JSON can include them. The 'names'
    # attribute is a dict like {0: 'Glass', 1: 'Metal', ...}.
    class_names_map = getattr(val_results, "names", {}) or {}
    class_names = [str(class_names_map.get(i, str(i))) for i in range(nc)]

    return reduced, class_names


def _compute_no_background_metrics(
    cm_no_bg: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute precision/recall/F1 from the confusion matrix once the background
    row and column have already been stripped.

    With background included, precision drops every time the model predicts
    an object that isn't there and recall drops every time the model misses
    a ground truth. With background excluded, the metrics answer a narrower
    question: 'given a prediction or ground truth that paired up with some
    real annotation, was the class correct?' This isolates pure class
    confusion errors from detection / localisation errors.
    """
    # Rows of 'cm_no_bg' are predicted classes; columns are actual classes.
    nc = cm_no_bg.shape[0]

    # --- Per-class precision / recall / F1 from the reduced matrix ---
    per_class: List[Dict[str, Any]] = []
    for i in range(nc):
        # True positives: predicted class i and actual class i.
        tp = float(cm_no_bg[i, i])
        # False positives in the no-background sense: predicted class i but
        # the actual class was a different real class (background ignored).
        fp = float(cm_no_bg[i, :].sum() - tp)
        # False negatives in the no-background sense: actual class i but
        # the prediction was a different real class.
        fn = float(cm_no_bg[:, i].sum() - tp)

        # Standard precision and recall, with guards against division by zero.
        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p > 0 else 0.0
        recall = tp / denom_r if denom_r > 0 else 0.0
        denom_f = precision + recall
        f1 = 2 * precision * recall / denom_f if denom_f > 0 else 0.0

        per_class.append({
            "class_id": int(i),
            "class_name": class_names[i],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # Total ground-truth instances for this class in the reduced
            # matrix. Useful context when reading per-class numbers.
            "support": int(cm_no_bg[:, i].sum()),
        })

    # --- Aggregate metrics across classes ---
    # Macro averages weight every class equally regardless of support size.
    macro_precision = sum(c["precision"] for c in per_class) / nc
    macro_recall = sum(c["recall"] for c in per_class) / nc
    macro_f1 = sum(c["f1"] for c in per_class) / nc

    # Overall classification accuracy on detected objects: diagonal mass over
    # the total mass of the reduced matrix. In the no-background case every
    # error is symmetric (each off-diagonal cell is both an FP for one class
    # and an FN for another), so micro precision = micro recall = F1, all
    # equal to this accuracy value.
    total_count = float(cm_no_bg.sum())
    diag_sum = float(np.diag(cm_no_bg).sum())
    accuracy = diag_sum / total_count if total_count > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        # Keep the raw reduced matrix in the report so downstream tools or a
        # human reader can recompute anything they want from it.
        "confusion_matrix_no_background": cm_no_bg.astype(int).tolist(),
        "class_names": class_names,
    }


def _save_no_background_confusion_matrix_plots(
    cm_no_bg: np.ndarray,
    class_names: List[str],
    save_dir: Path,
) -> List[Path]:
    """
    Save two PNG heatmaps for the no-background confusion matrix:

      'confusion_matrix_no_background.png'             - raw integer counts
      'confusion_matrix_no_background_normalized.png'  - per-actual-class proportions

    Plots are written next to Ultralytics' own 'confusion_matrix.png' so the
    background-included and background-excluded versions sit side by side
    in the same run folder. Returns the list of files actually written.
    Plotting is best-effort: if matplotlib is missing or any draw call
    raises, the function logs a warning and returns an empty list rather
    than breaking the rest of the evaluation pipeline.
    """
    # Import matplotlib lazily so its absence does not break callers that
    # never request a plot. Ultralytics already depends on matplotlib in
    # this project, so the import should normally succeed.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Warning: matplotlib not available, skipping no-bg plot: {exc}")
        return []

    nc = cm_no_bg.shape[0]
    if nc < 1:
        return []

    output_paths: List[Path] = []

    # Helper that performs the actual heatmap rendering for one variant.
    def _draw_one(display: np.ndarray, normalize: bool) -> Path:
        # Scale the figure with the number of classes so labels stay readable.
        fig, ax = plt.subplots(
            figsize=(max(6.0, nc * 1.4), max(5.0, nc * 1.2)),
            tight_layout=True,
        )
        # 'imshow' draws the matrix as a coloured grid; 'Blues' matches the
        # Ultralytics colour scheme so the two plots feel consistent.
        im = ax.imshow(display, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax)

        # Tick labels: one per class on both axes.
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("True class")
        ax.set_ylabel("Predicted class")

        title = "Confusion Matrix (no background)"
        if normalize:
            title += " - normalized"
        ax.set_title(title)

        # Annotate each cell with its value. Text colour flips to white on
        # darker cells so it stays readable.
        threshold = float(display.max()) / 2.0 if display.size > 0 else 0.0
        for row in range(nc):
            for col in range(nc):
                value = float(display[row, col])
                if normalize:
                    cell_text = f"{value:.2f}"
                else:
                    cell_text = f"{int(cm_no_bg[row, col])}"
                cell_color = "white" if value > threshold else "black"
                ax.text(col, row, cell_text, ha="center", va="center", color=cell_color)

        # File name mirrors the Ultralytics convention so the two pairs of
        # plots sit next to each other in the run folder.
        suffix = "_normalized" if normalize else ""
        out_path = Path(save_dir) / f"confusion_matrix_no_background{suffix}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    try:
        # --- Raw count plot ---
        output_paths.append(_draw_one(cm_no_bg.astype(float), normalize=False))

        # --- Normalized plot ---
        # Normalise per actual class so each column sums to 1 (when that
        # class has at least one ground truth in this split). Columns with
        # zero ground truth are left at zero rather than producing NaN.
        normalized = cm_no_bg.astype(float)
        col_sums = normalized.sum(axis=0, keepdims=True)
        non_zero_cols = col_sums.flatten() > 0
        if non_zero_cols.any():
            normalized[:, non_zero_cols] = (
                normalized[:, non_zero_cols] / col_sums[:, non_zero_cols]
            )
        output_paths.append(_draw_one(normalized, normalize=True))
    except Exception as exc:
        # Plotting is purely diagnostic; a failure here must not break the run.
        print(f"Warning: failed to save no-background confusion matrix plot: {exc}")

    return output_paths


def _extract_metrics(
    val_results: Any,
    split: str,
    *,
    include_no_background: bool = False,
) -> Dict[str, Any]:
    """
    Pull standard detection metrics out of an Ultralytics validation result object.

    Returns a nested dict with the split name, aggregate metrics, and per-class
    precision/recall/F1 values. When 'include_no_background' is True a
    'no_background_metrics' key is added carrying classification metrics
    computed from the confusion matrix with background excluded.
    """
    box = val_results.box

    # Collect aggregate metrics.
    metrics = {
        "precision": _safe_float(box.mp),
        "recall": _safe_float(box.mr),
        "f1": _mean_f1(box),
        "map50": _safe_float(box.map50),
        "map75": _safe_float(getattr(box, "map75", None)),
        "map50_95": _safe_float(box.map),
    }

    # Collect per-class precision and recall if class names are available.
    per_class: List[Dict[str, Any]] = []
    class_names = getattr(val_results, "names", {})
    try:
        for idx, name in class_names.items():
            per_class.append({
                "class_id": int(idx),
                "class_name": str(name),
                "precision": _safe_float(float(np.asarray(box.p)[idx])),
                "recall": _safe_float(float(np.asarray(box.r)[idx])),
            })
    except Exception:
        # Per-class data is best-effort; a failure here should not break evaluation.
        pass

    result: Dict[str, Any] = {
        "split": split,
        "num_classes": int(box.nc) if hasattr(box, "nc") else None,
        "metrics": metrics,
        "per_class": per_class,
        "ultralytics_save_dir": str(val_results.save_dir)
            if hasattr(val_results, "save_dir") else None,
    }

    # Optionally enrich the result with no-background classification metrics
    # and write companion confusion-matrix plots next to Ultralytics' own.
    if include_no_background:
        reduced = _reduce_confusion_matrix(val_results)
        if reduced is None:
            # Confusion matrix not available: record None so the user can see
            # that the option was honoured even though no data was produced.
            result["no_background_metrics"] = None
        else:
            cm_no_bg, class_names = reduced
            no_bg = _compute_no_background_metrics(cm_no_bg, class_names)

            # Save the no-background PNG heatmaps inside the same Ultralytics
            # run folder as 'confusion_matrix.png' for an easy side-by-side
            # comparison.
            save_dir = getattr(val_results, "save_dir", None)
            if save_dir is not None:
                plot_paths = _save_no_background_confusion_matrix_plots(
                    cm_no_bg, class_names, Path(save_dir)
                )
                no_bg["plot_paths"] = [str(p) for p in plot_paths]

            result["no_background_metrics"] = no_bg

    return result


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def run_split_evaluation(
    model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    run_name: str,
    *,
    plots: bool = True,
    exclude_background: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate 'model' on one dataset split (e.g. 'val' or 'test').

    When 'exclude_background' is True, the returned dict also contains a
    'no_background_metrics' field with classification metrics computed from
    the confusion matrix excluding the background class.
    """
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        plots=plots,
        save_json=True,
    )
    return _extract_metrics(
        val_results,
        split,
        include_no_background=exclude_background,
    )


def evaluate_checkpoint(
    weights: Path,
    data_yaml: Path,
    *,
    splits: List[str] = ("val", "test"),
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    eval_project: str = "runs/evaluate",
    run_name_prefix: str = "eval",
    plots: bool = True,
    exclude_background: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a trained checkpoint on one or more dataset splits.

    Loads the model once and runs validation for each requested split.
    Returns a summary dict containing the weights path and per-split results.
    When 'exclude_background' is True, each split result also includes
    classification metrics computed from the confusion matrix with the
    background row/column dropped.
    """
    # Load the checkpoint once to avoid reloading for each split.
    model = YOLO(str(weights))

    results_per_split: Dict[str, Any] = {}
    for split in splits:
        # Use a separate subfolder per split so Ultralytics does not overwrite outputs.
        run_name = f"{run_name_prefix}_{split}"
        results_per_split[split] = run_split_evaluation(
            model=model,
            data_yaml=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=eval_project,
            run_name=run_name,
            plots=plots,
            exclude_background=exclude_background,
        )

        # Print a short summary after each split with the standard metrics.
        m = results_per_split[split]["metrics"]
        print(
            f"[{split}] "
            f"P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"mAP50={m['map50']:.4f}  "
            f"mAP50-95={m['map50_95']:.4f}"
        )

        # If no-background metrics were requested, print a second line so the
        # user can immediately compare them with the standard metrics above.
        if exclude_background:
            no_bg = results_per_split[split].get("no_background_metrics")
            if no_bg is None:
                print(f"[{split} no-bg] confusion matrix not available; skipping.")
            else:
                print(
                    f"[{split} no-bg] "
                    f"accuracy={no_bg['accuracy']:.4f}  "
                    f"macroP={no_bg['macro_precision']:.4f}  "
                    f"macroR={no_bg['macro_recall']:.4f}  "
                    f"macroF1={no_bg['macro_f1']:.4f}"
                )

    return {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "exclude_background": exclude_background,
        "results": results_per_split,
    }


def save_evaluation_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write the evaluation summary dictionary to a JSON file."""
    write_json(output_path, summary)
    print(f"Evaluation report saved to: {output_path}")
