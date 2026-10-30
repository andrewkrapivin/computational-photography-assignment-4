import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage.io import imsave

from cp_hw2 import lRGB2XYZ

EPSILON = 1e-9
LENSLET_SIZE = 16

def readimg(path, dtype=np.double):
    # returns image in [0,1]
    # cv2 instead of skimage since it reads the 11200x6400 lightfield PNG much faster
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    assert img is not None, f"could not read {path}"
    img = img[:, :, ::-1]   # BGR -> RGB
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(dtype) / np.iinfo(img.dtype).max
    return img.astype(dtype)

def saveimg(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, (np.clip(img, 0, 1) * 255).round().astype(np.uint8))
    print(f"Saved image to {path}")

def gamma_encode(img):
    is_lin = (img <= 0.0031308)
    lin_part = 12.92 * img
    non_lin_part = (1.055) * np.pow(img, 1/2.4) - 0.055
    return lin_part * is_lin + non_lin_part * (1-is_lin)

def gamma_decode(img):
    # inverse of gamma_encode (sRGB -> linear)
    is_lin = (img <= 0.0404482)
    lin_part = img / 12.92
    non_lin_part = np.pow((img + 0.055) / 1.055, 2.4)
    return lin_part * is_lin + non_lin_part * (1-is_lin)

def luminance(lRGB):
    # Y channel of XYZ, same as assignments 2 and 3
    return lRGB2XYZ(lRGB)[:,:,1]


## Lightfield and sub-aperture views

def show_image(img, title, **kwargs):
    fig, ax = plt.subplots()
    ax.imshow(img, **kwargs)
    ax.set_title(title)
    plt.show()
    plt.close(fig)

def plot_lenslet(L, s, t):
    show_image(L[:, :, s, t, :], f"Lenslet (s, t) = ({s}, {t})", interpolation="nearest")

def plot_subaperture(L, u, v):
    show_image(L[u, v, :, :, :], f"Sub-aperture (u, v) = ({u}, {v})")

def load_lightfield(img, lenslet_size=LENSLET_SIZE):
    l = lenslet_size
    lx = img.shape[0] // l
    ly = img.shape[1] // l
    res = img[:lx * l, :ly * l].reshape(lx, l, ly, l, 3).transpose(1, 3, 0, 2, 4)
    # plot_lenslet(res, 0, 0)
    # plot_subaperture(res, 0, 0)
    return res

def tile_grid(grid):
    """Arrange a grid of images grid[i, j] (shape (I, J, H, W, 3)) into one (I*H, J*W, 3) image,
    with i increasing downward and j to the right. Used for the sub-aperture mosaic and the
    focal-aperture collage."""
    # TODO (Sub-aperture views)
    l, m = grid.shape[0], grid.shape[1]
    w,h = grid.shape[2], grid.shape[3]
    tiles = grid.transpose(0,2,1,3,4).reshape(w*l, h*m, 3)
    # show_image(tiles[:4*w, :4*h,:], "Part of the tiled grid")
    return tiles


## Refocusing and focal stacks
def uv_offset(lenslet_size=LENSLET_SIZE):
    return (lenslet_size - 1) / 2

def aperture_mask(a, shape="square", lenslet_size=LENSLET_SIZE):
    u = np.arange(lenslet_size) - uv_offset(lenslet_size)
    v = u[None, :]
    u = u[:, None] 

    if shape == "square":
        if a == None:
            a = lenslet_size // 2
        return (np.abs(u) <= a / 2) & (np.abs(v) <= a / 2) 
    elif shape == "circle":
        if a == None:
            a = lenslet_size
        return u**2 + v**2 <= (a / 2)**2
    elif shape == "heart":
        if a == None:
            a = lenslet_size * 3 // 4
        x = v * 2.3 / a
        y = -u * 2.3 / a + 0.12
        return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0


def shift_image(img, du, dv):
    h, w = img.shape[:2]
    iu, iv = int(np.floor(du)), int(np.floor(dv))
    fu, fv = du - iu, dv - iv

    def int_shift(a, b):   # img sampled at (s + a, t + b) for integers a, b
        rows = np.clip(np.arange(h) + a, 0, h - 1)
        cols = np.clip(np.arange(w) + b, 0, w - 1)
        return img[rows][:, cols]

    return ((1 - fu) * (1 - fv) * int_shift(iu, iv)     + (1 - fu) * fv * int_shift(iu, iv + 1)
          + fu       * (1 - fv) * int_shift(iu + 1, iv) + fu       * fv * int_shift(iu + 1, iv + 1))


def refocus(L, d, A, lenslet_size=LENSLET_SIZE):
    """Eq. (2)"""
    s = np.zeros_like(L[0,0,:,:,:])
    center_coords = uv_offset(lenslet_size)
    for (du,dv) in np.argwhere(A):
        # the chessboard lightfield's v axis is mirrored relative to u (views move down with u but
        # left with v), so the t shift takes the opposite sign of Eq. (2)
        shifted = shift_image(L[du, dv], (du-center_coords)*d, -(dv-center_coords)*d)
        s += shifted
    return s / np.count_nonzero(A)

def focal_stack(L, ds, A):
    """Stack of refocused images, shape (len(ds), S, T, 3)."""
    return np.stack([refocus(L, d, A) for d in ds])


## All-in-focus image and depth from focus

def sharpness_weights(stack, sigma1, sigma2):
    """Eqs. (3)-(6): w_sharpness(s, t, d), shape (len(stack), S, T), shared by the color channels."""
    lum = np.stack([luminance(gamma_decode(img)) for img in stack])
    # blur only within each image, not across depths
    low = gaussian_filter(lum, (0, sigma1, sigma1))
    high = lum - low
    return gaussian_filter(high**2, (0, sigma2, sigma2))

def all_in_focus(stack, w):
    """Eq. (7)."""
    return (w[..., None] * stack).sum(0) / (w.sum(0)[..., None] + EPSILON)

def depth_from_focus(w, ds):
    """Eq. (8)."""
    return (w * ds[:, None, None]).sum(0) / (w.sum(0) + EPSILON)


## Focal-aperture stack and confocal stereo

def focal_aperture_stack(L, ds, apertures, shape):
    """Shape (len(apertures), len(ds), S, T, 3); row i is the focal stack for aperture apertures[i].
    float32 to keep it in memory: 8 apertures x 41 depths is ~1.1 GB."""
    return np.stack([focal_stack(L, ds, aperture_mask(a, shape)).astype(np.float32) for a in apertures])

def afi(fa_stack, s, t):
    """Aperture-focus image of pixel (s, t): rows are apertures, columns focus settings, shape (A, D, 3)."""
    return fa_stack[:, :, s, t]

def confocal_depth(fa_stack, ds, apertures, shape, sigma=0):
    """Hasinoff and Kutulakos Eqs. (7) and (8): per-pixel depth from the AFIs.
    E_xy for each aperture is the number of (u, v) pairs in A. refocus() already divides by |A|,
    so do not normalize again here. sigma > 0 blurs the per-pixel cost spatially before the argmin."""
    # Eq. (7): in focus, the exitance-normalized intensity is the same for every aperture, so score
    # each focus setting by the variance across apertures (summed over color channels)
    var = np.zeros(fa_stack.shape[1:4], dtype=np.float32)
    for c in range(fa_stack.shape[-1]):   # per channel to keep the float32 stack from being copied
        var += fa_stack[..., c].var(axis=0)
    if sigma > 0:
        var = gaussian_filter(var, (0, sigma, sigma))
    # Eq. (8): depth is the focus setting with the most consistent AFI column
    return np.asarray(ds)[np.argmin(var, axis=0)]


## Pipelines

def depth_to_img(depth, ds):
    # [d_min, d_max] -> [0, 1] for saving
    return (depth - ds.min()) / (ds.max() - ds.min())

def aperture_tag(a, shape):
    return f"{shape}_afull" if a is None else f"{shape}_a{a:g}"

def stack_tag(args):
    return f"d{args.d_min}_{args.d_max}_n{args.n_d}_{aperture_tag(args.aperture, args.aperture_shape)}"

def load_or_compute_stack(L, ds, args):
    # the focal stack is reused by every --sigma1/--sigma2 run, so cache it next to the outputs
    path = Path(args.out).parent / "stacks" / f"stack_{stack_tag(args)}.npy"
    if args.skip_stack and path.is_file():
        print(f"Loading focal stack from {path}")
        return np.load(path)
    stack = focal_stack(L, ds, aperture_mask(args.aperture, args.aperture_shape)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stack)
    return stack

def mosaic_pipeline(L, args):
    out = Path(args.out)
    saveimg(tile_grid(L), out / "mosaic.png")
    U, V = L.shape[:2]
    saveimg(L[U // 2, V // 2], out / "center_view.png")

def refocus_pipeline(L, ds, args):
    # one folder per aperture, e.g. results/refocus/circle_a8/
    out = Path(args.out) / aperture_tag(args.aperture, args.aperture_shape)
    stack = load_or_compute_stack(L, ds, args)
    for d, img in zip(ds, stack):
        saveimg(img, out / f"refocus_d{d:.2f}.png")

def allfocus_pipeline(L, ds, args):
    out = Path(args.out)
    stack = load_or_compute_stack(L, ds, args)
    w = sharpness_weights(stack, args.sigma1, args.sigma2)
    tag = f"s1-{args.sigma1}_s2-{args.sigma2}_{stack_tag(args)}"
    saveimg(all_in_focus(stack, w), out / f"allfocus_{tag}.png")
    depth = depth_from_focus(w, ds)
    np.save(out / f"depth_{tag}.npy", depth)
    saveimg(depth_to_img(depth, ds), out / f"depth_{tag}.png")

def confocal_pipeline(L, ds, args):
    out = Path(args.out)
    fa_stack = focal_aperture_stack(L, ds, args.apertures, args.aperture_shape)
    tag = f"d{args.d_min}_{args.d_max}_n{args.n_d}_{args.aperture_shape}"
    # collage: every collage_step-th depth, so the tiles stay readable
    saveimg(tile_grid(fa_stack[:, ::args.collage_step]), out / f"collage_{tag}.png")
    for s, t in args.afi_pixels:
        # nearest-neighbor upsampling so each (aperture, focus) cell is visible
        img = np.repeat(np.repeat(afi(fa_stack, s, t), args.afi_scale, axis=0), args.afi_scale, axis=1)
        saveimg(img, out / f"afi_s{s}_t{t}_{tag}.png")
    depth = confocal_depth(fa_stack, ds, args.apertures, args.aperture_shape)
    np.save(out / f"depth_confocal_{tag}.npy", depth)
    saveimg(depth_to_img(depth, ds), out / f"depth_confocal_{tag}.png")


def main(argv=None):
    all_methods = ["mosaic", "refocus", "allfocus", "confocal"]
    parser = argparse.ArgumentParser(description="Lightfield rendering, depth from focus, and confocal stereo.")
    parser.add_argument("--method", default="mosaic", choices=all_methods,
                        help="mosaic: sub-aperture views; refocus: focal stack; allfocus: all-in-focus image and "
                             "depth from focus; confocal: focal-aperture stack and confocal stereo")
    parser.add_argument("--lightfield", default="data/chessboard_lightfield.png", help="Lenslet image path")
    parser.add_argument("--out", help="Output folder (default results/<method>)")
    # refocusing
    parser.add_argument("--d_min", type=float, default=-2.0, help="Smallest focus depth d")
    parser.add_argument("--d_max", type=float, default=2.0, help="Largest focus depth d")
    parser.add_argument("--n_d", type=int, default=41, help="Number of depths in the focal stack")
    parser.add_argument("--aperture", type=float, default=None,
                        help="Aperture side (square) or diameter (circle) in (u, v) units, default the largest")
    parser.add_argument("--aperture_shape", default="square", choices=["square", "circle", "heart"])
    parser.add_argument("--skip_stack", action="store_true", help="Load the cached focal stack instead of refocusing")
    # depth from focus
    parser.add_argument("--sigma1", type=float, default=2.0, help="sigma_1 of the low-pass blur, Eq. (4)")
    parser.add_argument("--sigma2", type=float, default=4.0, help="sigma_2 of the sharpness blur, Eq. (6)")
    # confocal stereo
    parser.add_argument("--apertures", type=float, nargs="+", default=[2, 4, 6, 8, 10, 12, 14, 16],
                        help="Aperture sizes of the focal-aperture stack")
    parser.add_argument("--collage_step", type=int, default=4, help="Show every n-th depth in the collage")
    parser.add_argument("--afi_pixels", type=lambda p: tuple(int(x) for x in p.split(",")), nargs="*",
                        default=[], help="Pixels s,t to save AFIs for, e.g. --afi_pixels 250,100 60,400")
    parser.add_argument("--afi_scale", type=int, default=16, help="Upsampling factor of the saved AFIs")
    args = parser.parse_args(argv)

    args.out = args.out or f"results/{args.method}"
    ds = np.linspace(args.d_min, args.d_max, args.n_d)

    L = load_lightfield(readimg(args.lightfield, dtype=np.float32))
    print(L.dtype, L.shape)

    if args.method == "mosaic":
        mosaic_pipeline(L, args)
    elif args.method == "refocus":
        refocus_pipeline(L, ds, args)
    elif args.method == "allfocus":
        allfocus_pipeline(L, ds, args)
    elif args.method == "confocal":
        confocal_pipeline(L, ds, args)


if __name__ == "__main__":
    main()
