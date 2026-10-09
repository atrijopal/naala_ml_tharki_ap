# splitsnap/preprocess.py
"""preprocess.py - EXIF fix, optional receipt crop + perspective warp, pre-model quality gate.
POLICY: whatever you choose here must be applied identically at training and inference time.
Test 'EXIF only' vs 'EXIF + crop' on your real validation bills and keep the better one."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(src):
    img = src if isinstance(src, Image.Image) else Image.open(src)
    return ImageOps.exif_transpose(img).convert("RGB")      # phone photos are often rotated via metadata only


MAX_SIDE = 2560      # EDA: SROIE has 35 MP scans, CORD up to 12 MP; augmenting those would be slow and memory-hungry


def cap_size(img, max_side=MAX_SIDE):
    if max(img.size) <= max_side:
        return img
    s = max_side / max(img.size)
    return img.resize((max(int(img.width * s), 1), max(int(img.height * s), 1)), Image.LANCZOS)


def load_image_fast(src, target=1280):
    """Inference-only loader. A JPEG can be decoded at 1/2, 1/4 or 1/8 scale (PIL `draft`), which skips most of the work for a
    12-megapixel phone photo. We only ask for a scale that keeps BOTH sides >= target (the model input is at most 1280 px on its long
    side), so the later single resize in the Donut processor still downsamples. Non-JPEG input and PIL images take the normal path."""
    if isinstance(src, Image.Image):
        return ImageOps.exif_transpose(src).convert("RGB")
    img = Image.open(src)
    if img.format == "JPEG":
        img.draft("RGB", (target, target))
    return ImageOps.exif_transpose(img).convert("RGB")


def prepare_image(src, use_crop=False, enhance=None, max_side=MAX_SIDE, fast=False):
    """THE single entry point used by training, validation, evaluation AND serving.
    enhance: None (default) or '+'-joined filter names from ENHANCERS, e.g. 'shadow+clahe'.
    Filters are EXPERIMENTS: keep one only if it wins on real validation bills, then use it everywhere."""
    img = load_image_fast(src) if fast else cap_size(load_image(src), max_side)
    if use_crop:
        img, _found = quad_crop(img)
    return enhance_image(img, enhance) if enhance else img


def _bgr(img):
    return np.array(img)[:, :, ::-1].copy()


def _rgb(arr):
    return Image.fromarray(arr[:, :, ::-1].copy())


def f_clahe(img, clip=2.0, tile=8):
    """Local contrast boost on the lightness channel (helps faded thermal print)."""
    lab = cv2.cvtColor(_bgr(img), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[:, :, 0])
    return _rgb(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def f_shadow(img):
    """Shadow / uneven-light removal: divide each channel by an estimate of its background."""
    arr, out = _bgr(img), []
    for ch in cv2.split(arr):
        bg = cv2.medianBlur(cv2.dilate(ch, np.ones((7, 7), np.uint8)), 21)
        out.append(cv2.normalize(255 - cv2.absdiff(ch, bg), None, 0, 255, cv2.NORM_MINMAX))
    return _rgb(cv2.merge(out))


def f_denoise(img, h=5):
    return _rgb(cv2.fastNlMeansDenoisingColored(_bgr(img), None, h, h, 7, 21))


def f_sharpen(img, amount=0.6, sigma=1.2):
    arr = _bgr(img); blur = cv2.GaussianBlur(arr, (0, 0), sigma)
    return _rgb(cv2.addWeighted(arr, 1 + amount, blur, -amount, 0))


def estimate_skew(img, max_deg=10.0, step=0.5):
    """Tilt in degrees (positive = rotate the image by +angle to straighten it): the angle whose row-projection of
    dark pixels is sharpest. Coarse (0.5 deg grid) and only reliable for text-dominated images."""
    g = np.array(img.convert("L")); s = 600.0 / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s) if s < 1 else g
    bw = (g < cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]).astype(np.uint8)
    h, w = bw.shape; best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + step, step):
        r = cv2.warpAffine(bw, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h))
        v = float(np.var(r.sum(axis=1)))
        if v > best:
            best, best_a = v, float(a)
    return best_a


def f_deskew(img, max_deg=10.0, step=0.5):
    best_a = estimate_skew(img, max_deg, step)
    if abs(best_a) < 0.4:
        return img
    arr = _bgr(img); H, W = arr.shape[:2]
    fill = tuple(int(x) for x in np.median(arr.reshape(-1, 3), axis=0))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), best_a, 1.0)
    return _rgb(cv2.warpAffine(arr, M, (W, H), borderValue=fill, flags=cv2.INTER_CUBIC))


ENHANCERS = {"clahe": f_clahe, "shadow": f_shadow, "denoise": f_denoise, "sharpen": f_sharpen, "deskew": f_deskew}


def enhance_image(img, spec):
    for name in str(spec).split("+"):
        img = ENHANCERS[name](img)
    return img


def _order_points(pts):
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def _four_point(arr, pts):
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    return cv2.warpPerspective(arr, cv2.getPerspectiveTransform(pts, dst), (w, h))


def quad_crop(img, min_area_frac=0.25):
    """Find the biggest 4-corner contour and flatten it. Returns (image, found_flag)."""
    arr = np.array(img)[:, :, ::-1].copy()
    h, w = arr.shape[:2]
    s = 800.0 / max(h, w)
    small = cv2.resize(arr, (max(int(w * s), 1), max(int(h * s), 1)))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area >= min_area_frac * small.shape[0] * small.shape[1] and area > best_area:
            best, best_area = approx.reshape(4, 2) / s, area
    if best is None:
        return img, False
    warped = _four_point(arr, _order_points(best.astype("float32")))
    if min(warped.shape[:2]) < 100:
        return img, False
    return Image.fromarray(warped[:, :, ::-1]), True


def quality_report(img, blur_min=15.0, dark=60.0, bright=225.0, contrast_min=40.0, washed_contrast=60.0, min_long_side=640):
    """Cheap pre-model checks that drive the 'please retake' prompt (R2 bonus).
    Re-based on the EDA (2026-10-09). The old gate (blur 60, std 25, long side 900) flagged 55% of CORD photos and called
    19% of clean SROIE scans 'washed out' (std is tiny on mostly-white pages). Now: sharpness floor = CORD's 5th percentile
    (15), and contrast = paper (median gray) minus ink (1st percentile gray), which does not depend on how much of the page
    is text. PROVISIONAL: thresholds are data-informed guesses; confirm on real phone photos before relying on them."""
    g = np.array(img.convert("L"))
    h, w = g.shape
    g = cv2.resize(g, (1000, max(int(h * 1000 / max(w, 1)), 1)))
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean, std = float(g.mean()), float(g.std())
    paper, ink = float(np.percentile(g, 50)), float(np.percentile(g, 1))
    contrast = paper - ink
    problems = []
    if max(h, w) < min_long_side:
        problems.append("low_resolution")
    if blur < blur_min:
        problems.append("blurry")
    if mean < dark:
        problems.append("too_dark")
    if paper > bright and contrast < washed_contrast:
        problems.append("washed_out")
    elif contrast < contrast_min:
        problems.append("low_contrast")
    return {"blur_var": round(blur, 1), "mean": round(mean, 1), "std": round(std, 1), "contrast": round(contrast, 1),
            "problems": problems, "retake": bool(problems)}
