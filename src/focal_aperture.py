import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image
from scipy.ndimage import gaussian_filter

from main import (saveimg, gamma_encode, gamma_decode, luminance, tile_grid, sharpness_weights,
                  all_in_focus, depth_from_focus, confocal_depth, depth_to_img)

EXIF_IFD = 0x8769
EXPOSURE_TIME = 33434
F_NUMBER = 33437


## Loading

def list_captures(folder):
    """{f-number: [paths sorted by focus index]} for files named f<N>_<index>.nef (or .jpg),
    as saved by take_focal_stack.py. NEFs are preferred since they are linear."""
    captures = {}
    for path in sorted(Path(folder).iterdir()):
        m = re.fullmatch(r"f([\d.]+)_(\d+)\.(nef|jpg)", path.name, re.IGNORECASE)
        if m:
            captures.setdefault(float(m[1]), {}).setdefault(int(m[2]), []).append(path)
    files = {}
    for n, shots in captures.items():
        files[n] = [min(shots[i], key=lambda p: p.suffix.lower() != ".nef") for i in sorted(shots)]
    return files

def exposure(path):
    """t / N^2 from the EXIF of the shot's JPEG (PIL does not read NEF EXIF)."""
    jpg = path.with_suffix(".jpg")
    exif = Image.open(jpg if jpg.exists() else path).getexif().get_ifd(EXIF_IFD)
    return float(exif[EXPOSURE_TIME]) / float(exif[F_NUMBER]) ** 2

def read_linear(path, width):
    """Linear RGB in [0, 1], resized to the given width."""
    if path.suffix.lower() == ".nef":
        with rawpy.imread(str(path)) as raw:
            img = raw.postprocess(half_size=True, gamma=(1, 1), no_auto_bright=True, output_bps=16,
                                  use_camera_wb=True) / 65535.0
    else:
        img = gamma_decode(cv2.imread(str(path))[:, :, ::-1] / 255.0)
    h = round(img.shape[0] * width / img.shape[1])
    return cv2.resize(img.astype(np.float32), (width, h), interpolation=cv2.INTER_AREA)

def load_fa_stack(folder, width):
    """Focal-aperture stack, shape (A, F, H, W, 3), rows sorted from the largest aperture (smallest
    f-number) down. Each image is divided by its t / N^2, so all apertures share one radiometric scale."""
    files = list_captures(folder)
    n_focus = {len(paths) for paths in files.values()}
    assert len(n_focus) == 1, f"apertures have different numbers of focus settings: {n_focus}"
    f_numbers = sorted(files)
    rows = []
    for n in f_numbers:
        print(f"Reading f/{n:g}: {len(files[n])} images")
        rows.append(np.stack([read_linear(p, width) / exposure(p) for p in files[n]]))
    return np.stack(rows), f_numbers


## Radiometric calibration and alignment

def normalize_exitance(fa_stack, sigma):
    """Divide out what t / N^2 misses: per-pixel exitance differences between apertures, mostly
    vignetting. Each aperture's average over focus settings, heavily blurred so defocus does not
    matter, is compared to the smallest aperture's (the last row). sigma = 0 uses a single gain.
    First, each shot is scaled to the same mean over the central half of the frame: defocus only
    spreads light around, so that mean should not change, but the captures jump by up to ~35% from
    shot to shot (lighting changes, or exposures off from their EXIF values)."""
    h, w = fa_stack.shape[2:4]
    center = fa_stack[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4].mean(axis=(2, 3, 4))
    print(f"per-shot gain: {(center / center.mean()).min():.3f}-{(center / center.mean()).max():.3f}")
    fa_stack *= (center.mean() / center)[:, :, None, None, None]
    ref = luminance(fa_stack[-1].mean(0))
    for a in range(len(fa_stack)):
        mean = luminance(fa_stack[a].mean(0))
        if sigma > 0:
            gain = gaussian_filter(mean, sigma) / gaussian_filter(ref, sigma)
        else:
            gain = np.full_like(mean, mean.mean() / ref.mean())
        print(f"aperture row {a}: exitance gain {gain.min():.3f}-{gain.max():.3f}")
        fa_stack[a] /= gain[None, :, :, None]
    return fa_stack

def similarity(warp, center):
    """The global scaling (plus translation) closest to an affine warp: same scale about the image
    center, no rotation or shear, as the magnification change from refocusing is."""
    s = np.sqrt(abs(np.linalg.det(warp[:, :2])))
    c = warp[:, :2] @ center + warp[:, 2]
    return np.float32([[s, 0, c[0] - s * center[0]], [0, s, c[1] - s * center[1]]])

def estimate_alignment(stack):
    """Per-focus-setting warps onto the middle focus setting, estimated with ECC on one aperture's
    focal stack (the smallest aperture, which is sharp at every focus setting). Works outward from
    the middle, initializing each with its neighbor's warp."""
    gray = [cv2.GaussianBlur(luminance(gamma_encode(np.clip(img / stack.max(), 0, 1))), (0, 0), 1.5)
            .astype(np.float32) for img in stack]
    n = len(stack)
    mid = n // 2
    h, w = gray[0].shape
    center = np.array([(w - 1) / 2, (h - 1) / 2])
    warps = [None] * n
    warps[mid] = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    for i in [*range(mid + 1, n), *range(mid - 1, -1, -1)]:
        init = warps[i - 1] if i > mid else warps[i + 1]
        try:
            _, warp = cv2.findTransformECC(gray[mid], gray[i], init.copy(), cv2.MOTION_AFFINE, criteria, None, 5)
            warps[i] = similarity(warp, center)
        except cv2.error:
            print(f"WARNING: ECC did not converge for focus setting {i}, reusing its neighbor's warp")
            warps[i] = init
        print(f"focus {i:3d}: scale {warps[i][0, 0]:.4f}, shift ({warps[i][0, 2]:+.1f}, {warps[i][1, 2]:+.1f})")
    return np.stack(warps)

def align(fa_stack, warps):
    """Warp every aperture's image at focus setting i with warps[i], then crop the border the warps
    leave undefined."""
    h, w = fa_stack.shape[2:4]
    for a in range(fa_stack.shape[0]):
        for i, warp in enumerate(warps):
            fa_stack[a, i] = cv2.warpAffine(fa_stack[a, i], warp, (w, h),
                                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                            borderMode=cv2.BORDER_REPLICATE)
    # where the corners of the output land in each input; anything outside the input is undefined
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], dtype=np.float32)
    mapped = np.concatenate([corners @ warp.T for warp in warps])
    margin = int(np.ceil(max(0, -mapped[:, 0].min(), mapped[:, 0].max() - (w - 1),
                             -mapped[:, 1].min(), mapped[:, 1].max() - (h - 1))))
    print(f"Cropping {margin} px of border")
    return fa_stack[:, :, margin:h - margin, margin:w - margin]


## Pipeline

def to_display(img, white):
    return gamma_encode(np.clip(img / white, 0, 1))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Depth from focus and confocal stereo on a captured focal-aperture stack.")
    parser.add_argument("--folder", required=True, help="Folder of f<N>_<index>.nef/.jpg captures from take_focal_stack.py")
    parser.add_argument("--out", help="Output folder (default results/focal_aperture/<folder name>)")
    parser.add_argument("--width", type=int, default=800, help="Resize images to this width first")
    parser.add_argument("--skip_load", action="store_true", help="Load the cached aligned stack instead of the captures")
    parser.add_argument("--no_align", action="store_true", help="Skip the magnification alignment")
    parser.add_argument("--exitance_sigma", type=float, default=40,
                        help="Blur (px) of the per-pixel exitance estimate; 0 uses one gain per aperture")
    parser.add_argument("--focus_range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Only use focus settings FIRST..LAST (inclusive), e.g. the ones where something is in focus")
    parser.add_argument("--sigma1", type=float, default=2.0, help="sigma_1 of the low-pass blur, Eq. (4)")
    parser.add_argument("--sigma2", type=float, default=4.0, help="sigma_2 of the sharpness blur, Eq. (6)")
    parser.add_argument("--confocal_sigma", type=float, default=2.0,
                        help="Blur of the confocal cost before the argmin; 0 is purely per-pixel")
    parser.add_argument("--collage_step", type=int, default=3, help="Show every n-th focus setting in the collage")
    parser.add_argument("--collage_width", type=int, default=200, help="Width of each collage tile")
    parser.add_argument("--afi_pixels", type=lambda p: tuple(int(x) for x in p.split(",")), nargs="*",
                        default=[], help="Pixels row,col (in the cropped output) to save AFIs for")
    parser.add_argument("--afi_scale", type=int, default=16, help="Upsampling factor of the saved AFIs")
    args = parser.parse_args(argv)

    out = Path(args.out or f"results/focal_aperture/{Path(args.folder).name}")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"w{args.width}_e{args.exitance_sigma:g}" + ("_noalign" if args.no_align else "")
    cache = out / f"fa_stack_{tag}.npz"

    if args.skip_load and cache.is_file():
        print(f"Loading focal-aperture stack from {cache}")
        data = np.load(cache)
        fa_stack, f_numbers = data["fa_stack"], list(data["f_numbers"])
    else:
        fa_stack, f_numbers = load_fa_stack(args.folder, args.width)
        fa_stack = normalize_exitance(fa_stack, args.exitance_sigma)
        if not args.no_align:
            warps = estimate_alignment(fa_stack[-1])
            np.save(out / f"warps_{tag}.npy", warps)
            fa_stack = align(fa_stack, warps)
        np.savez(cache, fa_stack=fa_stack, f_numbers=f_numbers)
    print(f"Focal-aperture stack {fa_stack.shape}, f-numbers {[f'{n:g}' for n in f_numbers]}")

    ds = np.arange(fa_stack.shape[1], dtype=np.float64)   # depth in units of focus setting index
    if args.focus_range:
        first, last = args.focus_range
        fa_stack, ds = fa_stack[:, first:last + 1], ds[first:last + 1]
        tag += f"_f{first}-{last}"
    n_focus = fa_stack.shape[1]
    white = np.percentile(fa_stack, 99.5)        # one display exposure for every image

    # example frames: first, middle and last focus setting at the largest and smallest apertures
    for a in (0, len(f_numbers) - 1):
        for i in (0, n_focus // 2, n_focus - 1):
            saveimg(to_display(fa_stack[a, i], white), out / f"frame_f{f_numbers[a]:g}_{int(ds[i]):03d}_{tag}.png")

    k = max(1, fa_stack.shape[3] // args.collage_width)
    saveimg(to_display(tile_grid(fa_stack[:, ::args.collage_step, ::k, ::k]), white),
            out / f"collage_step{args.collage_step}_{tag}.png")
    for s, t in args.afi_pixels:
        img = np.repeat(np.repeat(fa_stack[:, :, s, t], args.afi_scale, axis=0), args.afi_scale, axis=1)
        saveimg(to_display(img, white), out / f"afi_s{s}_t{t}_{tag}.png")

    # depth from focus and all-in-focus on the largest aperture; sharpness_weights expects sRGB
    stack = to_display(fa_stack[0], white)
    w = sharpness_weights(stack, args.sigma1, args.sigma2)
    dff_tag = f"s1-{args.sigma1}_s2-{args.sigma2}_{tag}"
    saveimg(all_in_focus(stack, w), out / f"allfocus_{dff_tag}.png")
    depth = depth_from_focus(w, ds)
    np.save(out / f"depth_{dff_tag}.npy", depth)
    saveimg(depth_to_img(depth, ds), out / f"depth_{dff_tag}.png")

    # confocal stereo on all apertures; the stack is already divided by each aperture's exitance
    depth = confocal_depth(fa_stack, ds, f_numbers, None, sigma=args.confocal_sigma)
    conf_tag = f"cs{args.confocal_sigma:g}_{tag}"
    np.save(out / f"depth_confocal_{conf_tag}.npy", depth)
    saveimg(depth_to_img(depth, ds), out / f"depth_confocal_{conf_tag}.png")


if __name__ == "__main__":
    main()
