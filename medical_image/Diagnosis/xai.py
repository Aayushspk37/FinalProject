# ============================================================
# X-HViT EXPLAINABILITY / HEATMAP GENERATION (ENHANCED)
# ============================================================
#
# Supports:
#   1. Grad-CAM++       — multi-stage, SmoothGrad, TTA, border trim
#   2. Global Attention Rollout — layer-skip + percentile clip
#   3. Window Attention Map     — spatially reconstructed
#   4. Hybrid Transformer       — global + window fusion
#   5. CNN + Transformer Agreement (geometric / adaptive)
#   6. Original-image overlays  — real RGB, variable alpha
#   7. Combined visualization   — 5-panel figure
#   8. Clinical 2-panel figure  — Original | AI Heatmap + colorbar
#   9. Deletion/Insertion AUC   — quantitative faithfulness
#
# Public function signatures unchanged from the original module.
# ============================================================

import os
import io
import base64
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors


# ============================================================
# CONFIGURATION
# ============================================================

HEATMAP_SIZE = 224

DEFAULT_ATTENTION_DISCARD_RATIO = 0.0
DEFAULT_BORDER_MARGIN = 0

DEFAULT_COMBINATION_MODE = "geometric"

DEFAULT_GRADCAMPP_STAGES = (5, 7)
DEFAULT_GRADCAMPP_STAGE_WEIGHTS = (0.4, 0.6)
DEFAULT_GRADCAMPP_N_SMOOTH = 8
DEFAULT_GRADCAMPP_NOISE_LEVEL = 0.08


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def _normalize_map(array):
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value + 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value + 1e-8)).astype(np.float32)


def _resize_map(array_2d, size=HEATMAP_SIZE):
    if array_2d is None:
        return None
    if array_2d.shape == (size, size):
        return array_2d.astype(np.float32)
    return cv2.resize(array_2d, (size, size),
                      interpolation=cv2.INTER_CUBIC).astype(np.float32)


# ============================================================
# ORIGINAL IMAGE PREPARATION
# ============================================================

def _prepare_original_image(image_file, target_size=224):
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image = PILImage.open(image_file).convert("RGB")
    image = image.resize((target_size, target_size), PILImage.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _tensor_to_display_image(img_tensor, target_size=224):
    image = img_tensor[0].detach().cpu().float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    image = image.astype(np.uint8)
    if image.shape[:2] != (target_size, target_size):
        image = cv2.resize(image, (target_size, target_size),
                           interpolation=cv2.INTER_LINEAR)
    return image


# ============================================================
# GRAD-CAM++ — MULTI-STAGE + SMOOTHGRAD
# ============================================================

def _compute_gradcampp_single_stage(model, img_tensor, class_idx,
                                    stage_idx, border_margin=1):
    """Grad-CAM++ on a single ConvNeXt stage. Returns [H,W] normalized or None."""
    model.eval()

    activations = {}
    gradients = {}

    target_layer = model.backbone.feature_extractor[stage_idx]

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        input_tensor = img_tensor.clone().detach().requires_grad_(True)
        logits = model(input_tensor)

        if logits.ndim != 2:
            return None

        target_score = logits[:, class_idx].sum()
        target_score.backward()

        if "value" not in activations or "value" not in gradients:
            return None

        acts = activations["value"]
        grads = gradients["value"]

        if acts.ndim != 4 or grads.ndim != 4:
            return None

        acts = acts[0].float()
        grads = grads[0].float()
        eps = 1e-8

        grads_sq = grads.pow(2)
        grads_cube = grads.pow(3)
        spatial_term = (acts * grads_cube).sum(dim=(1, 2), keepdim=True)
        denominator = 2.0 * grads_sq + spatial_term
        alpha = grads_sq / (denominator.abs() + eps)
        positive_grads = F.relu(grads)
        weights = (alpha * positive_grads).sum(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam).detach().cpu().numpy()

        if border_margin > 0 and cam.ndim == 2 and min(cam.shape) > 2 * border_margin:
            cam[:border_margin, :] = 0
            cam[-border_margin:, :] = 0
            cam[:, :border_margin] = 0
            cam[:, -border_margin:] = 0

        return _normalize_map(cam)

    except Exception as e:
        print(f"Grad-CAM++ single-stage failed (stage={stage_idx}):", e)
        return None

    finally:
        forward_handle.remove()
        backward_handle.remove()


def _compute_gradcampp_array(model, img_tensor, class_idx, border_margin=1):
    """
    Multi-stage SmoothGrad-CAM++.
    Averages over stages (5, 7) with weights (0.4, 0.6), plus 8 noisy copies.
    """
    model.eval()
    accum = None
    weight_sum = 0.0

    for i in range(DEFAULT_GRADCAMPP_N_SMOOTH):
        noisy = img_tensor if i == 0 else \
                img_tensor + DEFAULT_GRADCAMPP_NOISE_LEVEL * torch.randn_like(img_tensor)

        for stage_idx, weight in zip(DEFAULT_GRADCAMPP_STAGES,
                                     DEFAULT_GRADCAMPP_STAGE_WEIGHTS):
            cam = _compute_gradcampp_single_stage(model, noisy, class_idx,
                                                  stage_idx, border_margin)
            if cam is None:
                continue
            cam = _resize_map(cam, HEATMAP_SIZE)
            accum = weight * cam if accum is None else accum + weight * cam
            weight_sum += weight

    if accum is None or weight_sum == 0:
        return None

    return _normalize_map(accum / weight_sum)


# ============================================================
# GLOBAL ATTENTION ROLLOUT — ENHANCED
# ============================================================

def _compute_global_attention_rollout(model, img_tensor,
                                      skip_first_n_layers=1,
                                      percentile_clip=80):
    model.eval()

    with torch.no_grad():
        _ = model(img_tensor)

    global_attentions, _ = model.get_attentions()
    if not global_attentions:
        raise RuntimeError("No global attention weights captured.")

    result = None
    for layer_idx, attention in enumerate(global_attentions):
        if attention is None:
            continue
        if layer_idx < skip_first_n_layers:
            continue

        if attention.ndim == 4:
            attention = attention.mean(dim=1)
        if attention.ndim != 3:
            raise RuntimeError(f"Unexpected attention shape: {tuple(attention.shape)}")

        attention = attention[0].detach().float().cpu()
        n_tokens = attention.shape[-1]
        identity = torch.eye(n_tokens, dtype=attention.dtype)
        attention = attention + identity
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)

        result = attention if result is None else attention @ result

    if result is None:
        raise RuntimeError("Global attention rollout produced no result.")

    cls_to_patch = result[0, 1:]
    expected_tokens = model.spatial_h * model.spatial_w

    if cls_to_patch.numel() != expected_tokens:
        raise RuntimeError(
            f"Global attention token mismatch: got {cls_to_patch.numel()}, "
            f"expected {expected_tokens}"
        )

    attention_map = cls_to_patch.reshape(model.spatial_h, model.spatial_w).numpy()
    attention_map = _normalize_map(attention_map)

    if percentile_clip:
        thresh = np.percentile(attention_map, percentile_clip)
        attention_map = np.where(attention_map >= thresh, attention_map, 0)
        attention_map = _normalize_map(attention_map)

    up = cv2.resize(attention_map, (HEATMAP_SIZE, HEATMAP_SIZE),
                    interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 4)
    return _normalize_map(up)


# ============================================================
# WINDOW ATTENTION MAP
# ============================================================

def _compute_window_attention_map(model, img_tensor):
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)

    _, window_attentions = model.get_attentions()
    if not window_attentions:
        raise RuntimeError("No window attention weights captured.")

    spatial_h = model.spatial_h
    spatial_w = model.spatial_w

    total_map = None
    count = 0

    for attention in window_attentions:
        if attention is None or attention.ndim != 4:
            continue

        attention = attention.detach().float().cpu()
        attention = attention.mean(dim=1)
        token_importance = attention.mean(dim=1)

        window_size = int(model.encoder[0].window_attn.window_size)
        padded_h = int(np.ceil(spatial_h / window_size)) * window_size
        padded_w = int(np.ceil(spatial_w / window_size)) * window_size
        expected_windows = (padded_h // window_size) * (padded_w // window_size)

        if token_importance.shape[0] != expected_windows:
            continue

        local = token_importance.reshape(
            padded_h // window_size, padded_w // window_size,
            window_size, window_size
        ).permute(0, 2, 1, 3).reshape(padded_h, padded_w)
        local = local[:spatial_h, :spatial_w]

        total_map = local if total_map is None else total_map + local
        count += 1

    if total_map is None or count == 0:
        raise RuntimeError("Window attention could not be converted to a spatial map.")

    total_map = (total_map / float(count)).numpy()
    up = cv2.resize(_normalize_map(total_map), (HEATMAP_SIZE, HEATMAP_SIZE),
                    interpolation=cv2.INTER_CUBIC)
    return _normalize_map(up)


# ============================================================
# HYBRID TRANSFORMER MAP
# ============================================================

def _compute_hybrid_transformer_map(model, img_tensor):
    global_map = _compute_global_attention_rollout(model, img_tensor)
    window_map = _compute_window_attention_map(model, img_tensor)

    if global_map.shape != window_map.shape:
        window_map = cv2.resize(window_map,
                                (global_map.shape[1], global_map.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    hybrid = 0.5 * global_map + 0.5 * window_map
    return _normalize_map(hybrid)


# ============================================================
# COMBINATION  (geometric / product / mean / adaptive)
# ============================================================

def _combine_explanations(cam, transformer_map, mode="geometric"):
    if cam is None or transformer_map is None:
        return None

    cam = _normalize_map(cam)
    transformer_map = _normalize_map(transformer_map)

    if cam.shape != transformer_map.shape:
        transformer_map = cv2.resize(transformer_map,
                                     (cam.shape[1], cam.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

    if mode == "geometric":
        combined = np.sqrt(np.clip(cam, 0, 1) * np.clip(transformer_map, 0, 1))
    elif mode == "product":
        combined = np.clip(cam, 0, 1) * np.clip(transformer_map, 0, 1)
    elif mode == "mean":
        combined = 0.5 * cam + 0.5 * transformer_map
    elif mode == "adaptive":
        try:
            from scipy.spatial.distance import jensenshannon
            f1 = cam.flatten() + 1e-8
            f2 = transformer_map.flatten() + 1e-8
            f1 /= f1.sum()
            f2 /= f2.sum()
            agreement = float(np.clip(1 - jensenshannon(f1, f2), 0, 1))
        except Exception:
            agreement = 0.5
        w_cam = 0.5 + 0.2 * (1 - agreement)
        w_tf = 1.0 - w_cam
        combined = w_cam * cam + w_tf * transformer_map
    else:
        raise ValueError(f"Unknown combination mode: {mode}")

    return _normalize_map(combined)


# ============================================================
# RESIZE + OVERLAY  (variable alpha)
# ============================================================

def _resize_explanation(array_2d, target_size=224):
    if array_2d is None:
        return None
    resized = cv2.resize(array_2d, (target_size, target_size),
                         interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0, 1).astype(np.float32)


def _colorize_overlay(image_rgb, array_2d,
                      colormap=cv2.COLORMAP_JET,
                      alpha=0.45,
                      isolate_percentile=None,
                      gamma=0.85):
    if image_rgb is None or array_2d is None:
        return None

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    explanation = _resize_explanation(array_2d, image_rgb.shape[0])

    if isolate_percentile is not None:
        thresh = np.percentile(explanation, isolate_percentile)
        explanation = np.where(explanation > thresh, explanation, 0.0)
        explanation = _normalize_map(explanation)

    if gamma != 1.0:
        explanation = np.power(np.clip(explanation, 0, 1), gamma)

    heat_uint8 = (explanation * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    if isolate_percentile is not None:
        alpha_map = np.clip((explanation - 0.15) / 0.4, 0, 1)[..., None] * alpha
        overlay = (image_rgb * (1 - alpha_map) +
                   heat_rgb * alpha_map).astype(np.uint8)
    else:
        overlay = cv2.addWeighted(image_rgb, 1.0 - alpha,
                                  heat_rgb, alpha, 0)

    return overlay


# ============================================================
# TITLE + BASE64
# ============================================================

def _add_title(image, title):
    h, w = image.shape[:2]
    title_height = 34
    canvas = np.zeros((h + title_height, w, 3), dtype=np.uint8)
    canvas[title_height:, :] = image
    cv2.putText(canvas, title, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _image_to_base64(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode heatmap PNG.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/png;base64," + encoded


# ============================================================
# PUBLIC GENERATORS
# ============================================================

def generate_gradcam_heatmap(model, img_tensor, class_idx,
                             disease_key=None, original_image=None,
                             border_margin=0):
    try:
        if original_image is None:
            original_image = _tensor_to_display_image(img_tensor, HEATMAP_SIZE)

        cam = _compute_gradcampp_array(model, img_tensor, class_idx, border_margin)
        if cam is None:
            return None

        overlay = _colorize_overlay(original_image, cam, cv2.COLORMAP_JET,
                                    alpha=0.60, isolate_percentile=60, gamma=0.85)
        if overlay is None:
            return None
        return _image_to_base64(overlay)

    except Exception as e:
        print("Grad-CAM++ generation failed:", e)
        traceback.print_exc()
        return None


def generate_attention_map(model, img_tensor, disease_key=None, original_image=None):
    try:
        if original_image is None:
            original_image = _tensor_to_display_image(img_tensor, HEATMAP_SIZE)

        attention = _compute_global_attention_rollout(model, img_tensor)
        overlay = _colorize_overlay(original_image, attention,
                                    cv2.COLORMAP_VIRIDIS,
                                    alpha=0.60, isolate_percentile=60, gamma=0.85)
        if overlay is None:
            return None
        return _image_to_base64(overlay)

    except Exception as e:
        print("Global attention generation failed:", e)
        traceback.print_exc()
        return None


def generate_transformer_heatmap(model, img_tensor, disease_key=None, original_image=None):
    try:
        if original_image is None:
            original_image = _tensor_to_display_image(img_tensor, HEATMAP_SIZE)

        transformer_map = _compute_hybrid_transformer_map(model, img_tensor)
        overlay = _colorize_overlay(original_image, transformer_map,
                                    cv2.COLORMAP_VIRIDIS,
                                    alpha=0.60, isolate_percentile=60, gamma=0.85)
        if overlay is None:
            return None
        return _image_to_base64(overlay)

    except Exception as e:
        print("Transformer heatmap generation failed:", e)
        traceback.print_exc()
        return None


def generate_combined_heatmap(model, img_tensor, class_idx,
                              disease_key=None, original_image=None,
                              border_margin=0, combination_mode="geometric"):
    try:
        if original_image is None:
            original_image = _tensor_to_display_image(img_tensor, HEATMAP_SIZE)

        cam = _compute_gradcampp_array(model, img_tensor, class_idx, border_margin)
        if cam is None:
            raise RuntimeError("Grad-CAM++ failed.")

        transformer_map = _compute_hybrid_transformer_map(model, img_tensor)
        if transformer_map is None:
            raise RuntimeError("Transformer explanation failed.")

        combined = _combine_explanations(cam, transformer_map, mode=combination_mode)
        if combined is None:
            raise RuntimeError("Failed to combine explanations.")

        overlay = _colorize_overlay(original_image, combined,
                                    cv2.COLORMAP_JET,
                                    alpha=0.65, isolate_percentile=65, gamma=0.85)
        if overlay is None:
            return None
        return _image_to_base64(overlay)

    except Exception as e:
        print("Combined heatmap generation failed:", e)
        traceback.print_exc()
        return None


def generate_all_heatmaps(model, img_tensor, class_idx, disease_key=None,
                          original_image=None, border_margin=0,
                          combination_mode="geometric"):
    try:
        if original_image is None:
            original_image = _tensor_to_display_image(img_tensor, HEATMAP_SIZE)

        cam = _compute_gradcampp_array(model, img_tensor, class_idx, border_margin)
        global_attention = _compute_global_attention_rollout(model, img_tensor)
        window_attention = _compute_window_attention_map(model, img_tensor)
        transformer = _compute_hybrid_transformer_map(model, img_tensor)
        combined = _combine_explanations(cam, transformer, mode=combination_mode)

        def _colorize(array, colormap):
            if array is None:
                return None
            return _colorize_overlay(original_image, array, colormap,
                                     alpha=0.60, isolate_percentile=60, gamma=0.85)

        def _encode(image):
            return None if image is None else _image_to_base64(image)

        return {
            "gradcam":           _encode(_colorize(cam, cv2.COLORMAP_JET)),
            "global_attention":  _encode(_colorize(global_attention, cv2.COLORMAP_VIRIDIS)),
            "window_attention":  _encode(_colorize(window_attention, cv2.COLORMAP_VIRIDIS)),
            "transformer":       _encode(_colorize(transformer, cv2.COLORMAP_VIRIDIS)),
            "combined":          _encode(_colorize(combined, cv2.COLORMAP_JET)),
            "raw": {
                "gradcam":           cam,
                "global_attention":  global_attention,
                "window_attention":  window_attention,
                "transformer":       transformer,
                "combined":          combined,
            }
        }

    except Exception as e:
        print("Complete X-HViT explainability generation failed:", e)
        traceback.print_exc()
        return {
            "gradcam": None, "global_attention": None, "window_attention": None,
            "transformer": None, "combined": None,
            "raw": {"gradcam": None, "global_attention": None,
                    "window_attention": None, "transformer": None,
                    "combined": None}
        }


def generate_explainability_panel(model, img_tensor, class_idx,
                                  disease_key=None, original_image=None,
                                  border_margin=0,
                                  combination_mode="geometric"):
    try:
        if original_image is None:
            original_image = _tensor_to_display_image(img_tensor, HEATMAP_SIZE)

        cam = _compute_gradcampp_array(model, img_tensor, class_idx, border_margin)
        global_attention = _compute_global_attention_rollout(model, img_tensor)
        transformer = _compute_hybrid_transformer_map(model, img_tensor)
        combined = _combine_explanations(cam, transformer, mode=combination_mode)

        def _overlay_with_var_alpha(array, colormap):
            if array is None:
                return None
            return _colorize_overlay(original_image, array, colormap,
                                     alpha=0.60, isolate_percentile=60, gamma=0.85)

        gradcam_overlay     = _overlay_with_var_alpha(cam, cv2.COLORMAP_JET)
        global_overlay      = _overlay_with_var_alpha(global_attention, cv2.COLORMAP_VIRIDIS)
        transformer_overlay = _overlay_with_var_alpha(transformer, cv2.COLORMAP_VIRIDIS)
        combined_overlay    = _overlay_with_var_alpha(combined, cv2.COLORMAP_JET)

        original_panel    = _add_title(original_image, "Original")
        gradcam_panel     = _add_title(gradcam_overlay, "Grad-CAM++")
        global_panel      = _add_title(global_overlay, "Global Attention")
        transformer_panel = _add_title(transformer_overlay, "Hybrid Transformer")
        combined_panel    = _add_title(combined_overlay, "CNN + Transformer Agreement")

        panel = np.hstack([original_panel, gradcam_panel, global_panel,
                           transformer_panel, combined_panel])
        return _image_to_base64(panel)

    except Exception as e:
        print("Explainability panel generation failed:", e)
        traceback.print_exc()
        return None


def generate_heatmaps_for_prediction(model, img_tensor,
                                     disease_key=None,
                                     original_image=None,
                                     class_names=None):
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    class_idx = int(predicted_class.item())
    confidence_value = float(confidence.item())

    class_name = str(class_idx)
    if class_names is not None and (0 <= class_idx < len(class_names)):
        class_name = class_names[class_idx]

    heatmaps = generate_all_heatmaps(
        model=model, img_tensor=img_tensor, class_idx=class_idx,
        disease_key=disease_key, original_image=original_image,
        border_margin=0, combination_mode="geometric"
    )

    panel = generate_explainability_panel(
        model=model, img_tensor=img_tensor, class_idx=class_idx,
        disease_key=disease_key, original_image=original_image,
        border_margin=0, combination_mode="geometric"
    )

    return {
        "class_idx": class_idx,
        "class_name": class_name,
        "confidence": confidence_value,
        "heatmaps": heatmaps,
        "panel": panel,
    }


# ============================================================
# CLINICAL 2-PANEL — ORIGINAL | AI HEATMAP + COLORBAR
# ============================================================

def _clinical_gradcampp_single(model, x, class_idx, stage_idx, border_margin=1):
    """Single-stage Grad-CAM++ for clinical pipeline (self-contained)."""
    feats, grads = {}, {}
    layer = model.backbone.feature_extractor[stage_idx]
    fh = layer.register_forward_hook(lambda m, i, o: feats.update(v=o))
    bh = layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))
    try:
        x = x.clone().detach().requires_grad_(True)
        model.zero_grad()
        out = model(x)
        oh = torch.zeros_like(out); oh[0, class_idx] = 1
        out.backward(gradient=oh, retain_graph=True)

        if 'v' not in feats or 'v' not in grads:
            return None

        A, G = feats['v'][0], grads['v'][0]
        g2, g3 = G ** 2, G ** 3
        sa = A.sum(dim=(1, 2), keepdim=True)
        denom = 2 * g2 + sa * g3
        denom = torch.where(denom != 0, denom, torch.full_like(denom, 1e-8))
        alpha = g2 / denom
        w = (alpha * F.relu(G)).sum(dim=(1, 2))
        cam = F.relu((w[:, None, None] * A).sum(0)).detach().cpu().numpy()

        if border_margin > 0 and cam.ndim == 2 and min(cam.shape) > 2 * border_margin:
            cam[:border_margin, :]  = 0
            cam[-border_margin:, :] = 0
            cam[:, :border_margin]  = 0
            cam[:, -border_margin:] = 0

        return _normalize_map(cam)
    finally:
        fh.remove(); bh.remove()


def _clinical_compute_cam(model, img_tensor, class_idx,
                          stages=(5, 7), n_smooth=8, noise_level=0.08):
    model.eval()
    accum, count = None, 0

    for i in range(n_smooth):
        noisy = img_tensor if i == 0 else \
                img_tensor + noise_level * torch.randn_like(img_tensor)
        for si in stages:
            cam = _clinical_gradcampp_single(model, noisy, class_idx, si)
            if cam is None:
                continue
            cam = cv2.resize(cam.astype(np.float32), (224, 224),
                             interpolation=cv2.INTER_LINEAR)
            accum = cam.astype(np.float64) if accum is None else accum + cam
            count += 1

    if accum is None or count == 0:
        return None
    return _normalize_map(accum / count)


def _render_clinical_figure(gray_img, cam,
                            pred_label=None, confidence=None,
                            agreement=None,
                            target_size=600,
                            isolate_pct=65,
                            blur_sigma=8,
                            gamma=0.85,
                            dpi=140):
    """
    Build the matplotlib 2-panel clinical figure.
    Returns a BytesIO buffer containing the PNG.
    """
    img_disp = cv2.resize(gray_img, (target_size, target_size),
                          interpolation=cv2.INTER_LANCZOS4)

    cam_up = cv2.resize(cam, (target_size, target_size),
                        interpolation=cv2.INTER_CUBIC)
    cam_up = cv2.GaussianBlur(cam_up, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    thresh = np.percentile(cam_up, isolate_pct)
    cam_up = np.where(cam_up > thresh, cam_up, 0.0)
    cam_up = _normalize_map(cam_up)
    cam_up = np.power(cam_up, gamma)

    peak_y, peak_x = np.unravel_index(np.argmax(cam_up), cam_up.shape)

    fig = plt.figure(figsize=(16, 7), facecolor='black')

    # ── Left panel: original with red arrow ──
    ax0 = fig.add_axes([0.02, 0.06, 0.44, 0.88])
    ax0.imshow(img_disp, cmap='gray')
    ax0.axis('off')
    ax0.set_title('Original X-ray', color='white', fontsize=15, fontweight='bold',
                  pad=12,
                  bbox=dict(boxstyle='round,pad=0.5',
                            fc='#16233f', ec='#3a6ea5', alpha=0.95))

    tail_x = peak_x - 90 if peak_x > 90 else peak_x + 90
    tail_y = peak_y - 90 if peak_y > 90 else peak_y + 90
    ax0.annotate('', xy=(peak_x, peak_y), xytext=(tail_x, tail_y),
                 arrowprops=dict(arrowstyle='->', color='red',
                                 lw=3.0, shrinkA=0, shrinkB=6))

    # ── Right panel: bone-navy base + jet overlay ──
    ax1 = fig.add_axes([0.50, 0.06, 0.44, 0.88])

    bone_navy = plt.cm.bone(img_disp / 255.0)[..., :3]
    bone_navy = bone_navy * np.array([0.30, 0.42, 0.72])

    jet = plt.cm.jet(cam_up)[..., :3]
    alpha_mask = (cam_up ** 0.7)[..., None]
    blended = np.clip(bone_navy * (1 - alpha_mask) + jet * alpha_mask, 0, 1)

    ax1.imshow(blended)
    ax1.axis('off')
    ax1.set_title('AI Heatmap (Fracture Highlight)',
                  color='white', fontsize=15, fontweight='bold', pad=12,
                  bbox=dict(boxstyle='round,pad=0.5',
                            fc='#16233f', ec='#3a6ea5', alpha=0.95))

    caption = ''
    if pred_label and confidence is not None:
        caption = f'Pred: {pred_label} ({confidence:.1f}%)'
    if agreement is not None:
        caption += f'  |  Agreement: {agreement*100:.0f}%'
    if caption:
        ax1.text(0.5, -0.02, caption, transform=ax1.transAxes,
                 ha='center', va='top', color='white', fontsize=11,
                 path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # ── Colorbar ──
    cax = fig.add_axes([0.955, 0.10, 0.012, 0.80])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.05, 0.5, 0.95])
    cb.set_ticklabels(['Low\nsuspicion', 'Medium', 'High\nsuspicion'])
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.tick_params(labelsize=10, colors='white')
    plt.setp(cb.ax.get_yticklabels(), color='white')
    cb.outline.set_edgecolor('white')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_clinical_heatmap(model,
                              img_tensor,
                              class_idx=None,
                              disease_key=None,
                              original_image=None,
                              class_names=None,
                              confidence=None,
                              return_array=False,
                              stages=(5, 7),
                              n_smooth=8,
                              isolate_pct=65):
    """
    Generate the CLINICAL 2-PANEL heatmap: Original X-ray + AI Heatmap + colorbar.

    Returns:
      • base64 "data:image/png;base64,..."  (default)
      • OR (H,W,3) uint8 numpy array         (if return_array=True)
      • None on failure
    """
    try:
        model.eval()

        # ── 1. Determine class + confidence ──
        if class_idx is None or confidence is None:
            with torch.no_grad():
                probs = torch.softmax(model(img_tensor), dim=1)
                if class_idx is None:
                    class_idx = int(probs.argmax(dim=1).item())
                if confidence is None:
                    confidence = float(probs[0, class_idx].item())

        # ── 2. Resolve original image ──
        if original_image is None:
            arr = img_tensor[0].detach().cpu().float()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            arr = (arr * std + mean).clamp(0, 1)
            arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gray_img = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = original_image
            if gray_img.ndim == 3:
                gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
            if gray_img.dtype != np.uint8:
                gray_img = gray_img.astype(np.uint8)
            if gray_img.shape[:2] != (224, 224):
                gray_img = cv2.resize(gray_img, (224, 224),
                                      interpolation=cv2.INTER_LANCZOS4)

        # ── 3. Compute the CAM ──
        cam = _clinical_compute_cam(model, img_tensor, class_idx,
                                    stages=stages, n_smooth=n_smooth)
        if cam is None:
            print("[xai_clinical] CAM computation failed.")
            return None

        # ── 4. Label ──
        pred_label = None
        if class_names is not None and 0 <= class_idx < len(class_names):
            pred_label = class_names[class_idx]

        # ── 5. Render ──
        buf = _render_clinical_figure(
            gray_img, cam,
            pred_label=pred_label,
            confidence=(confidence * 100.0) if confidence is not None else None,
            agreement=None,
            isolate_pct=isolate_pct,
        )

        # ── 6. Encode ──
        if return_array:
            buf.seek(0)
            file_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"

    except Exception as e:
        print("[xai_clinical] Failed to generate clinical heatmap:", e)
        import traceback; traceback.print_exc()
        return None


def save_clinical_heatmap_to_file(model,
                                  img_tensor,
                                  out_path,
                                  class_idx=None,
                                  disease_key=None,
                                  original_image=None,
                                  class_names=None,
                                  confidence=None,
                                  isolate_pct=65):
    """Same as generate_clinical_heatmap but writes to disk. Returns True/False."""
    try:
        img = generate_clinical_heatmap(
            model=model, img_tensor=img_tensor, class_idx=class_idx,
            disease_key=disease_key, original_image=original_image,
            class_names=class_names, confidence=confidence,
            return_array=True, isolate_pct=isolate_pct,
        )
        if img is None:
            return False
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print("[xai_clinical] Failed to save heatmap:", e)
        return False


# ============================================================
# QUANTITATIVE FAITHFULNESS — Deletion / Insertion AUC
# ============================================================

def compute_deletion_insertion_auc(model, img_tensor, class_idx, cam,
                                   steps=15, device=None):
    """
    Insertion AUC + Deletion AUC for the given CAM.
    Lower deletion, higher insertion = more faithful explanation.
    """
    if device is None:
        device = img_tensor.device

    model.eval()
    cam_r = cv2.resize(cam.astype(np.float32), (224, 224),
                       interpolation=cv2.INTER_CUBIC)
    order = np.argsort(-cam_r.flatten())

    x_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    base_np = cv2.GaussianBlur(x_np, (31, 31), 0)
    base = torch.from_numpy(base_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    orig = img_tensor.clone()
    del_p, ins_p = [], []
    N = order.size

    with torch.no_grad():
        for k in range(1, steps + 1):
            n = int(N * (k / steps))
            idx = order[:n]

            # Deletion
            di = orig.clone()
            df = di.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            bf = base.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            df[idx] = bf[idx]
            di = df.reshape(1, 224, 224, 3).permute(0, 3, 1, 2)
            del_p.append(float(F.softmax(model(di), dim=1)[0, class_idx]))

            # Insertion
            ii = base.clone()
            inf_ = ii.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            of = orig.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            inf_[idx] = of[idx]
            ii = inf_.reshape(1, 224, 224, 3).permute(0, 3, 1, 2)
            ins_p.append(float(F.softmax(model(ii), dim=1)[0, class_idx]))

    xx = np.linspace(1 / steps, 1.0, steps)
    return float(np.trapz(del_p, xx)), float(np.trapz(ins_p, xx))