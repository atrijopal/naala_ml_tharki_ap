# splitsnap/augment.py
"""augment.py - on-the-fly training augmentation for hard bills (blur, tilt, faded print, shadows...).
No flips (they destroy text). No opaque occlusion over text (it creates unreadable-but-labelled
fields and teaches hallucination); shadows/edge cuts are used instead."""
import io, random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def _fill(img):
    return tuple(int(v) for v in np.median(np.array(img).reshape(-1, 3), axis=0))


def _rotate(img, rng, deg):
    return img.rotate(rng.uniform(-deg, deg), expand=True, fillcolor=_fill(img), resample=Image.BICUBIC)


def _perspective(img, rng, mag=0.04):
    a = np.array(img); h, w = a.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda m: rng.uniform(-m, m)
    dst = np.float32([[j(mag) * w, j(mag) * h], [w + j(mag) * w, j(mag) * h],
                      [w + j(mag) * w, h + j(mag) * h], [j(mag) * w, h + j(mag) * h]])
    out = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)


def _blur(img, rng):
    if rng.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 2.2)))
    k = rng.choice([3, 5, 7]); kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k                                  # horizontal motion blur
    ang = rng.uniform(0, 180)
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1)
    kern = cv2.warpAffine(kern, M, (k, k)); kern /= max(kern.sum(), 1e-6)
    return Image.fromarray(cv2.filter2D(np.array(img), -1, kern))


def _bc(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.25))
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))


def _faded(img, rng):                                           # faded thermal print
    a = np.array(img).astype(np.float32); alpha = rng.uniform(0.6, 0.85)
    return Image.fromarray(np.clip(255 - (255 - a) * alpha, 0, 255).astype(np.uint8))


def _shadow(img, rng):
    a = np.array(img).astype(np.float32); h, w = a.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = (xx * np.cos(ang) + yy * np.sin(ang)); ramp = (ramp - ramp.min()) / max(ramp.max() - ramp.min(), 1e-6)
    mask = 1.0 - rng.uniform(0.1, 0.45) * ramp
    return Image.fromarray(np.clip(a * mask[..., None], 0, 255).astype(np.uint8))


def _noise(img, rng):
    a = np.array(img).astype(np.float32)
    return Image.fromarray(np.clip(a + np.random.normal(0, rng.uniform(3, 12), a.shape), 0, 255).astype(np.uint8))


def _jpeg(img, rng):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.randint(25, 85)); buf.seek(0)
    return Image.open(buf).convert("RGB")


def _edge_cut(img, rng):                                        # receipt partly out of frame (<=3%)
    w, h = img.size; c = rng.uniform(0.0, 0.03)
    return img.crop((int(w * c * rng.random()), int(h * c * rng.random()), w - int(w * c * rng.random()), h - int(h * c * rng.random())))


def _noise_field(h, w, rng, scale=8):
    """Smooth random field in [0,1] (cheap value noise): small random grid upscaled with bicubic."""
    g = np.random.RandomState(rng.randrange(1 << 30)).rand(max(h // scale, 2), max(w // scale, 2)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _table_texture(w, h, rng):
    """Procedural table / cloth backgrounds (wood grain, checked cloth, plain, dark) - phone photos are rarely on flat colour."""
    kind = rng.choice(["wood", "wood", "cloth", "plain", "dark"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "wood":
        r0 = rng.randint(120, 205)                                   # real wood: red > green > blue
        base = np.array([r0, r0 * rng.uniform(0.55, 0.78), r0 * rng.uniform(0.25, 0.5)], np.float32)
        wob = _noise_field(h, w, rng, 40) * 4                     # gentle waviness, not swirls
        g = 0.5 + 0.5 * np.sin(yy * rng.uniform(0.08, 0.22) + wob + 1.2 * _noise_field(h, w, rng, 12))
        g = 0.7 * g + 0.3 * _noise_field(h, w, rng, 3)             # fine grain
        tex = base[None, None, :] * (0.78 + 0.35 * g[..., None])
    elif kind == "cloth":
        gray = rng.randint(60, 220)                                  # muted cloth: mix a random colour with grey
        c1 = 0.55 * np.array([rng.randint(30, 230) for _ in range(3)], np.float32) + 0.45 * gray; c2 = c1 * rng.uniform(0.72, 0.95)
        sz = rng.randint(14, 44); chk = (((xx // sz) + (yy // sz)) % 2)[..., None]
        tex = c1 * chk + c2 * (1 - chk)
        tex *= 0.9 + 0.15 * _noise_field(h, w, rng, 4)[..., None]
    elif kind == "plain":
        tex = np.array([rng.randint(150, 235)] * 3, np.float32)[None, None, :] + rng.randint(-12, 12)
    else:
        tex = np.array([rng.randint(25, 70)] * 3, np.float32)[None, None, :] + 0 * xx[..., None]
    tex = tex * (0.92 + 0.12 * _noise_field(h, w, rng, 20)[..., None])
    return Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))


def _photo_background(img, rng):
    """Receipt lying on a table: textured background + margin + soft contact shadow."""
    w, h = img.size; m = rng.randint(int(0.01 * max(w, h)), int(0.07 * max(w, h)))      # thin margin: receipt fills most of the frame
    bg = _table_texture(w + 2 * m, h + 2 * m, rng)
    sh = Image.new("L", bg.size, 0); sh.paste(120, (m + 6, m + 8, m + w + 6, m + h + 8))
    sh = sh.filter(ImageFilter.GaussianBlur(10))
    bg.paste(Image.new("RGB", bg.size, (0, 0, 0)), (0, 0), sh.point(lambda v: int(v * 0.55)))
    bg.paste(img, (m, m))
    return bg


def _crease(img, rng):
    """1-3 fold lines: a thin dark line with a lighter band beside it, like crumpled paper."""
    a = np.array(img).astype(np.float32); h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(rng.randint(1, 2)):
        th = rng.uniform(0, np.pi); c = (xx - rng.uniform(.1, .9) * w) * np.cos(th) + (yy - rng.uniform(.1, .9) * h) * np.sin(th)
        wd = rng.uniform(1.5, 4.0)
        prof = -0.09 * np.exp(-(c / wd) ** 2) + 0.05 * np.exp(-((c - 3 * wd) / (2.5 * wd)) ** 2)
        a *= (1 + prof * rng.uniform(.6, 1.4))[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _curl(img, rng):
    """Paper curl: smooth brightness bulge along one axis."""
    a = np.array(img).astype(np.float32); h, w = a.shape[:2]
    t = np.linspace(0, np.pi, w if rng.random() < .5 else h, dtype=np.float32)
    prof = 1 - rng.uniform(0.05, 0.2) * np.cos(t * rng.choice([1, 2]))
    m = prof[None, :, None] if len(prof) == w else prof[:, None, None]
    return Image.fromarray(np.clip(a * m, 0, 255).astype(np.uint8))


def _vignette_cast(img, rng):
    """Phone look: slight vignette + white-balance tint."""
    a = np.array(img).astype(np.float32); h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    a *= (1 - rng.uniform(0.08, 0.28) * np.clip(r - 0.4, 0, 1) ** 2)[..., None]
    a *= np.array([rng.uniform(.93, 1.06), rng.uniform(.95, 1.04), rng.uniform(.92, 1.07)], np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _lowres(img, rng):
    """Low-resolution capture / digital zoom: down then up."""
    w, h = img.size; s = rng.uniform(0.35, 0.7)
    return img.resize((max(int(w * s), 64), max(int(h * s), 64)), Image.BILINEAR).resize((w, h), Image.BICUBIC)


# probabilities per profile (EDA 2026-10-09: CORD already soft/dark phone photos -> gentle; SROIE flat sharp scans ->
# photo-style; synthetic bills are clean renders -> strongest, closest to real phone photos).
PROFILES = {
    "default": dict(bg=0.0, persp=.3, rot=.6, crease=0.0, curl=0.0, shadow=.3, vig=0.0, cut=.2, blur=.4, bc=.5, faded=.25, noise=.4, lowres=0.0, jpeg=.5, blur_sig=(.6, 2.2)),
    "cord":    dict(bg=0.0, persp=.2, rot=.5, crease=.1, curl=.1, shadow=.2, vig=.1, cut=.1, blur=.15, bc=.5, faded=.1, noise=.2, lowres=.05, jpeg=.3, blur_sig=(.4, 1.0)),
    "real":    dict(bg=0.0, persp=.2, rot=.5, crease=.1, curl=.1, shadow=.2, vig=.1, cut=.1, blur=.15, bc=.5, faded=.1, noise=.2, lowres=.05, jpeg=.3, blur_sig=(.4, 1.0)),
    "sroie":   dict(bg=.45, persp=.4, rot=.6, crease=.3, curl=.3, shadow=.5, vig=.4, cut=.15, blur=.4, bc=.6, faded=.2, noise=.4, lowres=.2, jpeg=.5, blur_sig=(.6, 2.0)),
    "synth":   dict(bg=.55, persp=.5, rot=.7, crease=.4, curl=.3, shadow=.5, vig=.4, cut=.15, blur=.4, bc=.6, faded=.2, noise=.5, lowres=.2, jpeg=.7, blur_sig=(.4, 1.2)),
}


def augment(img, rng=None, strength=1.0, profile="default"):
    """Training-only augmentation. profile in PROFILES (by data source). No flips, no opaque occlusion boxes."""
    rng = rng or random.Random(); P = PROFILES.get(profile, PROFILES["default"])
    if rng.random() < P["crease"]: img = _crease(img, rng)      # paper defects first, so they stay on the paper
    if rng.random() < P["curl"]: img = _curl(img, rng)
    if rng.random() < P["bg"]: img = _photo_background(img, rng)
    if rng.random() < P["persp"]: img = _perspective(img, rng, 0.04 * strength)
    if rng.random() < P["rot"]: img = _rotate(img, rng, 6 * strength)
    if rng.random() < P["shadow"]: img = _shadow(img, rng)
    if rng.random() < P["vig"]: img = _vignette_cast(img, rng)
    if rng.random() < P["cut"]: img = _edge_cut(img, rng)
    if rng.random() < P["blur"]:
        if rng.random() < 0.5: img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*P["blur_sig"])))
        else: img = _blur(img, rng)
    if rng.random() < P["bc"]: img = _bc(img, rng)
    if rng.random() < P["faded"]: img = _faded(img, rng)
    if rng.random() < P["noise"]: img = _noise(img, rng)
    if rng.random() < P["lowres"]: img = _lowres(img, rng)
    if rng.random() < P["jpeg"]: img = _jpeg(img, rng)
    return img
