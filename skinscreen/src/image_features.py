"""Image preprocessing and feature extraction.

The smartphone photos in PAD-UFES-20 come from different devices, with
different resolutions, lighting and angles (noise). Each image is

1. resized to IMG_SIZE x IMG_SIZE,
2. colour-normalised with the Shades-of-Gray colour-constancy algorithm
   (removes the colour cast of the light source),
3. lightly denoised (median filter),

and then described with hand-crafted colour, texture and shape features.
Optionally, deep embeddings from an ImageNet-pretrained ResNet-50 can be added
(``--cnn`` flag, requires torch + torchvision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageFilter
from scipy.stats import skew
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from . import config
from .utils import log

P = config.IMG_FEATURE_PREFIX


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def shades_of_gray(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi, 2004). img in [0,1]."""
    illum = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.linalg.norm(illum) + 1e-8)
    out = img / (illum * np.sqrt(3) + 1e-8)
    return np.clip(out, 0, 1)


def load_and_preprocess(path: str, size: int = config.IMG_SIZE) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        im = im.filter(ImageFilter.MedianFilter(3))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return shades_of_gray(arr)


# --------------------------------------------------------------------------- #
# Hand-crafted features
# --------------------------------------------------------------------------- #
def _color_moments(img: np.ndarray, space: str) -> dict:
    feats = {}
    for i in range(3):
        ch = img[..., i].ravel()
        feats[f"{P}{space}{i}_mean"] = float(ch.mean())
        feats[f"{P}{space}{i}_std"] = float(ch.std())
        feats[f"{P}{space}{i}_skew"] = float(skew(ch)) if ch.std() > 1e-8 else 0.0
    return feats


def _hsv_hist(hsv: np.ndarray) -> dict:
    feats = {}
    for i, (name, bins) in enumerate((("h", 8), ("s", 4), ("v", 4))):
        h, _ = np.histogram(hsv[..., i], bins=bins, range=(0, 1))
        h = h / h.sum()
        for b, v in enumerate(h):
            feats[f"{P}hist_{name}{b}"] = float(v)
    return feats


def _glcm(gray_u8: np.ndarray) -> dict:
    g = (gray_u8 // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(g, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    feats = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(glcm, prop)  # shape (distances, angles)
        for di, d in enumerate((1, 3)):
            feats[f"{P}glcm_{prop}_d{d}"] = float(vals[di].mean())
    return feats


def _lbp(gray_u8: np.ndarray, n_points: int = 8, radius: int = 1) -> dict:
    lbp = local_binary_pattern(gray_u8, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    hist = hist / hist.sum()
    return {f"{P}lbp_{i}": float(v) for i, v in enumerate(hist)}


def _lesion_shape(gray: np.ndarray, lab: np.ndarray) -> dict:
    """Rough lesion segmentation (Otsu: lesion is darker than skin)."""
    try:
        thr = threshold_otsu(gray)
    except ValueError:
        thr = gray.mean()
    mask = gray < thr
    feats = {f"{P}lesion_area": float(mask.mean())}
    lbl = label(mask)
    if lbl.max() > 0:
        props = max(regionprops(lbl), key=lambda r: r.area)
        per = props.perimeter if props.perimeter > 0 else 1.0
        feats[f"{P}lesion_eccentricity"] = float(props.eccentricity)
        feats[f"{P}lesion_solidity"] = float(props.solidity)
        feats[f"{P}lesion_extent"] = float(props.extent)
        feats[f"{P}lesion_compactness"] = float(4 * np.pi * props.area / per ** 2)
    else:
        for k in ("eccentricity", "solidity", "extent", "compactness"):
            feats[f"{P}lesion_{k}"] = 0.0
    inside, outside = mask, ~mask
    for i, ch in enumerate(("L", "a", "b")):
        a = lab[..., i]
        in_m = a[inside].mean() if inside.any() else a.mean()
        out_m = a[outside].mean() if outside.any() else a.mean()
        feats[f"{P}lesion_contrast_{ch}"] = float(in_m - out_m)
        feats[f"{P}lesion_std_{ch}"] = float(a[inside].std()) if inside.any() else 0.0
    return feats


def extract_features_one(path: str) -> dict:
    """All hand-crafted features + raw-image quality stats for one image."""
    with Image.open(path) as im:
        w, h = im.size
    img = load_and_preprocess(path)
    gray = rgb2gray(img)
    gray_u8 = (gray * 255).astype(np.uint8)
    hsv, lab = rgb2hsv(img), rgb2lab(img)

    feats = {}
    feats.update(_color_moments(img, "rgb"))
    feats.update(_color_moments(hsv, "hsv"))
    feats.update(_color_moments(lab / np.array([100.0, 128.0, 128.0]), "lab"))
    feats.update(_hsv_hist(hsv))
    feats.update(_glcm(gray_u8))
    feats.update(_lbp(gray_u8))
    feats.update(_lesion_shape(gray, lab))
    # quality stats used in EDA only (NOT model features: they describe the
    # phone/camera, not the lesion, and could act as a shortcut)
    feats["q_width"], feats["q_height"] = w, h
    feats["q_brightness"] = float(gray.mean())
    lap = np.abs(np.diff(gray, 2, axis=0)).mean() + np.abs(np.diff(gray, 2, axis=1)).mean()
    feats["q_sharpness"] = float(lap)
    return feats


def extract_handcrafted(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Return a DataFrame (index = img_id) of hand-crafted image features."""
    config.ensure_dirs()
    cache = config.CACHE_DIR / "image_features.csv"
    if use_cache and cache.exists():
        feats = pd.read_csv(cache, index_col="img_id")
        if set(df["img_id"]).issubset(feats.index):
            log(f"Loaded cached image features from {cache.name}")
            return feats.loc[df["img_id"]]

    log(f"Extracting hand-crafted features from {len(df)} images ...")
    rows = Parallel(n_jobs=config.N_JOBS, verbose=0)(
        delayed(extract_features_one)(p) for p in df["image_path"])
    feats = pd.DataFrame(rows, index=pd.Index(df["img_id"], name="img_id"))
    feats.to_csv(cache)
    log(f"  {feats.shape[1]} features per image (cached to {cache.name})")
    return feats


# --------------------------------------------------------------------------- #
# Optional deep features
# --------------------------------------------------------------------------- #
def extract_cnn_embeddings(df: pd.DataFrame, batch_size: int = 32,
                           use_cache: bool = True) -> pd.DataFrame:
    """2048-d global-average-pooled ResNet-50 (ImageNet) embeddings."""
    cache = config.CACHE_DIR / "cnn_features.csv"
    if use_cache and cache.exists():
        feats = pd.read_csv(cache, index_col="img_id")
        if set(df["img_id"]).issubset(feats.index):
            log("Loaded cached CNN embeddings")
            return feats.loc[df["img_id"]]
    try:
        import torch
        from torchvision import models, transforms
    except ImportError as e:
        raise ImportError("--cnn needs `pip install torch torchvision`") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    out = []
    paths = list(df["image_path"])
    log(f"Extracting ResNet-50 embeddings on {device} ...")
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([tf(Image.fromarray(
                (load_and_preprocess(p, 256) * 255).astype(np.uint8)))
                for p in paths[i:i + batch_size]]).to(device)
            out.append(net(batch).cpu().numpy())
    emb = np.vstack(out)
    feats = pd.DataFrame(emb, index=pd.Index(df["img_id"], name="img_id"),
                         columns=[f"{config.CNN_FEATURE_PREFIX}{i}" for i in range(emb.shape[1])])
    feats.to_csv(cache)
    return feats


def attach_image_features(df: pd.DataFrame, use_cnn: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Add image features to df. Returns (df, list_of_image_feature_columns).

    Columns starting with ``q_`` (resolution, brightness, sharpness) are kept
    for the EDA only and listed in ``df.attrs['quality_cols']``.
    """
    hand = extract_handcrafted(df)
    quality_cols = [c for c in hand.columns if c.startswith("q_")]
    feat_cols = [c for c in hand.columns if c.startswith(P)]
    df = df.join(hand, on="img_id")
    if use_cnn:
        cnn = extract_cnn_embeddings(df)
        df = df.join(cnn, on="img_id")
        feat_cols += list(cnn.columns)
    df.attrs["quality_cols"] = quality_cols
    return df, feat_cols
