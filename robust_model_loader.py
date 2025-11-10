# robust_model_loader.py
import torch
import json
import os

def inspect_checkpoint(path, top_n=60):
    """Return simple diagnostics: top keys and shapes for candidate weight tensors."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "net"):
            if k in ckpt:
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        return {"type": type(ckpt).__name__}
    summary = {}
    for k, v in ckpt.items():
        if hasattr(v, "shape"):
            summary[k] = tuple(v.shape)
    # return top keys
    items = list(summary.items())[:top_n]
    return dict(items)

def robust_load_model(model_cls, model_path, classes=None, device="cpu",
                      prefer_checkpoint_classifier=False, classes_json_save_path=None):
    """
    Safe loader:
    - prefers to detect final linear by matching weight tensors whose second dim==backbone_out_features
      (avoids matching backbone conv head which may have shape [out, in, kh, kw]).
    - Only copies checkpoint params whose names exist in model.state_dict() AND whose shapes match exactly.
    - Optionally recreates model with checkpoint classifier size if prefer_checkpoint_classifier=True.
    Returns: model, num_classes_used_for_model_init, info_dict
    """
    device = torch.device(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    raw = torch.load(model_path, map_location=device)
    # Unpack common wrappers
    if isinstance(raw, dict):
        for k in ("state_dict", "model_state_dict", "net"):
            if k in raw:
                raw = raw[k]
                break
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint does not contain a state dict mapping keys->tensors.")

    state_dict = raw

    # create a temp model to inspect backbone output features safely (we instantiate but freeze)
    # we instantiate model with num_classes=1 to get backbone feature size (EfficientNet b0 -> 1280)
    tmp_model = model_cls(num_classes=1, freeze_backbone=True)
    backbone_out_features = None
    try:
        # try to access backbone _fc in_features if available
        if hasattr(tmp_model, "backbone") and hasattr(tmp_model.backbone, "_fc"):
            backbone_out_features = tmp_model.backbone._fc.in_features
    except Exception:
        pass

    # fallback common default for efficientnet-b0
    if backbone_out_features is None:
        backbone_out_features = 1280

    # Heuristic: find checkpoint candidate classifier weight keys where
    # tensor.ndim == 2 and tensor.shape[1] == backbone_out_features
    candidate_classifier_keys = []
    for k, v in state_dict.items():
        if hasattr(v, "ndim") and v.ndim == 2:
            if v.shape[1] == backbone_out_features:
                candidate_classifier_keys.append((k, tuple(v.shape)))

    # choose best candidate: prefer keys that include "classifier", "fc", "head", "linear"
    chosen_ckpt_cls_key = None
    for pref in ("classifier", "classifier.1", "head", "fc", "linear", "classifier.weight"):
        for k, shape in candidate_classifier_keys:
            if pref in k:
                chosen_ckpt_cls_key = k
                break
        if chosen_ckpt_cls_key:
            break
    # if none matched preference but candidates exist, pick first
    if chosen_ckpt_cls_key is None and candidate_classifier_keys:
        chosen_ckpt_cls_key = candidate_classifier_keys[0][0]

    checkpoint_num_classes = None
    if chosen_ckpt_cls_key:
        checkpoint_num_classes = int(state_dict[chosen_ckpt_cls_key].shape[0])

    # Determine local desired num_classes
    local_num_classes = len(classes) if classes is not None else None
    if local_num_classes is None and checkpoint_num_classes is not None:
        # create stub classes
        local_num_classes = int(checkpoint_num_classes)
        if classes_json_save_path:
            try:
                stub = [f"class_{i}" for i in range(local_num_classes)]
                with open(classes_json_save_path, "w", encoding="utf-8") as f:
                    json.dump(stub, f, indent=2)
            except Exception:
                pass

    if local_num_classes is None:
        raise ValueError("Cannot determine number of classes. Provide 'classes' or use a checkpoint with classifier info.")

    # If user prefers checkpoint classifier and mismatch -> recreate model with checkpoint size
    if checkpoint_num_classes is not None and prefer_checkpoint_classifier and checkpoint_num_classes != local_num_classes:
        model = model_cls(num_classes=int(checkpoint_num_classes), freeze_backbone=True)
        model.to(device)
        # Only copy keys that exactly match name+shape
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        info = {
            "mode": "recreated_with_checkpoint_head",
            "chosen_checkpoint_classifier_key": chosen_ckpt_cls_key,
            "checkpoint_num_classes": checkpoint_num_classes,
            "loaded_keys": len(filtered),
            "missing_keys": missing,
            "unexpected_keys": unexpected
        }
        return model, int(checkpoint_num_classes), info

    # Otherwise: create local model with desired local_num_classes and only copy matching keys
    model = model_cls(num_classes=int(local_num_classes), freeze_backbone=True)
    model.to(device)

    model_state = model.state_dict()
    filtered = {}
    loaded_keys = []
    skipped_shape_mismatches = []
    skipped_missing_in_model = []

    for k, v in state_dict.items():
        if k in model_state:
            if tuple(model_state[k].shape) == tuple(v.shape):
                filtered[k] = v
                loaded_keys.append(k)
            else:
                skipped_shape_mismatches.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        else:
            skipped_missing_in_model.append((k, tuple(v.shape)))

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    info = {
        "mode": "local_head_kept",
        "chosen_checkpoint_classifier_key": chosen_ckpt_cls_key,
        "checkpoint_num_classes": checkpoint_num_classes,
        "local_num_classes": local_num_classes,
        "loaded_keys_count": len(loaded_keys),
        "skipped_shape_mismatches_count": len(skipped_shape_mismatches),
        "skipped_missing_in_model_count": len(skipped_missing_in_model),
        "skipped_shape_mismatches": skipped_shape_mismatches[:20],
        "skipped_missing_in_model": skipped_missing_in_model[:20],
        "missing_keys": missing,
        "unexpected_keys": unexpected
    }
    return model, int(local_num_classes), info
