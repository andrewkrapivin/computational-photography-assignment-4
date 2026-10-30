import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import correlate2d

from main import EPSILON, saveimg, gamma_decode, luminance

def read_frames(path, step=1, scale=1.0):
    """Every step-th video frame as RGB in [0, 1], resized by scale, shape (N, H, W, 3)."""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"could not open {path}"
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(frame[:, :, ::-1].astype(np.float32) / 255)   # BGR -> RGB; float32 to fit a few hundred frames
        i += 1
    cap.release()
    print(f"Read {len(frames)} of {i} frames from {path}")
    return np.stack(frames)

def gray(img):
    # template matching runs on luminance
    return luminance(gamma_decode(img))


## Template matching

def box_filter(img, size):
    """img correlated with a size x size box normalized to sum to 1 (I-bar in Eq. (9))."""
    # 'valid' mode: output [i, j] averages img[i:i+size, j:j+size], the patch under the template at (i, j).
    # The box is separable, so a row box then a column box gives the same result with far fewer operations.
    rows = correlate2d(img, np.ones((1, size)) / size, mode="valid")
    return correlate2d(rows, np.ones((size, 1)) / size, mode="valid")

def normalized_cross_correlation(g, I):
    """Eq. (9): h[i, j] for template g over image I, using only pixel-wise operations and correlate2d."""
    n = g.size
    g0 = g - g.mean()
    I_bar = box_filter(I, g.shape[0])
    # numerator: sum_kl g0[k,l] (I[i+k,j+l] - I_bar[i,j]) = (g0 correlated with I)[i,j] - I_bar[i,j] * sum(g0),
    # and sum(g0) = 0, so it is a single correlation
    num = correlate2d(I, g0, mode="valid")
    # sum_kl (I[i+k,j+l] - I_bar[i,j])^2 = n * (box(I^2) - I_bar^2)
    var_I = n * np.maximum(box_filter(I ** 2, g.shape[0]) - I_bar ** 2, 0)
    return num / np.sqrt(np.sum(g0 ** 2) * var_I + EPSILON)

def find_shift(h, ref):
    """Eq. (10): argmax of h, as the (row, col) shift that aligns this frame's match with ref,
    the template's position in the middle frame."""
    match = np.unravel_index(np.argmax(h), h.shape)
    return np.array(ref) - np.array(match)


## Refocusing

def shift_image(img, dy, dx):
    """img shifted by (dy, dx) pixels."""
    # shifts from Eq. (10) are integers, so this is a crop and paste; pixels shifted in from outside are 0
    H, W = img.shape[:2]
    out = np.zeros_like(img)
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        img[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out

def refocus(frames, row, col, size, window):
    """Template from the middle frame at (row, col) (top-left corner), size x size; match it in a
    window x window search window centered on it in every frame, shift each frame onto the
    middle frame, and average them. Returns the refocused image and the per-frame shifts."""
    N, H, W = frames.shape[:3]
    g = gray(frames[N // 2])[row:row + size, col:col + size]
    # search window centered on the template, clipped to the frame
    r_c, c_c = row + size // 2, col + size // 2
    r0, r1 = max(r_c - window // 2, 0), min(r_c + window // 2, H)
    c0, c1 = max(c_c - window // 2, 0), min(c_c + window // 2, W)
    ref = (row - r0, col - c0)   # template's top-left corner in window coordinates

    total = np.zeros(frames.shape[1:], np.float32)
    coverage = np.zeros((H, W, 1), np.float32)   # how many shifted frames cover each pixel
    ones = np.ones((H, W, 1), np.float32)
    shifts = np.zeros((N, 2), int)
    for t in range(N):
        h = normalized_cross_correlation(g, gray(frames[t, r0:r1, c0:c1]))
        shifts[t] = find_shift(h, ref)
        dy, dx = shifts[t]
        total += shift_image(frames[t], dy, dx)
        coverage += shift_image(ones, dy, dx)
        if t % 50 == 0:
            print(f"  frame {t}/{N}: shift ({dy}, {dx}), NCC {h.max():.2f}")
    print(f"  shift range: rows [{shifts[:, 0].min()}, {shifts[:, 0].max()}], cols [{shifts[:, 1].min()}, {shifts[:, 1].max()}]")
    return total / np.maximum(coverage, 1), shifts

def draw_box(img, row, col, size, color=(1.0, 0.0, 0.0), width=3):
    img = img.copy()
    cv2.rectangle(img, (col, row), (col + size, row + size), color, width)
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refocus an unstructured lightfield video by template matching.")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--out", default="results/unstructured", help="Output folder")
    parser.add_argument("--step", type=int, default=1, help="Use every n-th frame")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize frames by this factor first")
    parser.add_argument("--template", type=int, nargs=2, action="append", metavar=("ROW", "COL"), default=[],
                        help="Top-left corner of a template in the middle frame; repeat for several focus points")
    parser.add_argument("--size", type=int, default=60, help="Template side in pixels")
    parser.add_argument("--window", type=int, default=200, help="Search window side in pixels")
    args = parser.parse_args(argv)

    out = Path(args.out)
    frames = read_frames(args.video, args.step, args.scale)
    middle = frames[len(frames) // 2]
    # pick template corners from this frame (e.g. hover over it in an image viewer)
    saveimg(middle, out / f"middle_frame_step{args.step}_scale{args.scale}.png")

    for row, col in args.template:
        tag = f"r{row}_c{col}_t{args.size}_w{args.window}_step{args.step}_scale{args.scale}"
        result, shifts = refocus(frames, row, col, args.size, args.window)
        saveimg(result, out / f"refocus_{tag}.png")
        saveimg(draw_box(middle, row, col, args.size), out / f"template_{tag}.png")
        np.save(out / f"shifts_{tag}.npy", shifts)


if __name__ == "__main__":
    main()
