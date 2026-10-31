# Homework 4: Lightfields and focal stacks

All scripts live in `src/` and are run from the repository root. They need
`numpy`, `scipy`, `scikit-image`, `opencv-python`, `matplotlib`, `pillow` and `rawpy`,
plus `gphoto2` for capturing.

## 1. Lightfield rendering, depth from focus, and confocal stereo: `src/main.py`

`--method` picks the part of the assignment. Every method reads the lightfield
from `--lightfield` (default `data/chessboard_lightfield.png`) and writes to
`--out` (default `results/<method>`).

- `mosaic` (default): the 16x16 mosaic of sub-aperture views.
- `refocus`: refocuses at `--n_d` depths from `--d_min` to `--d_max` (Eq. 2)
  and saves every image of the focal stack.
- `allfocus`: the all-in-focus image and depth map from the focal stack (Eqs. 3-8).
- `confocal`: the focal-aperture stack, its collage, AFIs, and the confocal
  stereo depth map.

```
python src/main.py --method mosaic
python src/main.py --method refocus --d_min -2 --d_max 2 --n_d 41
python src/main.py --method allfocus --sigma1 2 --sigma2 4 --skip_stack
python src/main.py --method confocal --apertures 2 4 6 8 10 12 14 16 --afi_pixels 250,100
```

| Flag | Meaning | Default |
|---|---|---|
| `--d_min`, `--d_max`, `--n_d` | focus depths `np.linspace(d_min, d_max, n_d)` | `-2`, `2`, `41` |
| `--aperture` | aperture side (square) or diameter (circle) in `(u, v)` units | largest |
| `--aperture_shape` | `square` or `circle` | `square` |
| `--skip_stack` | load the focal stack cached in `results/stacks/` instead of refocusing | off |
| `--sigma1`, `--sigma2` | Gaussian sigmas of Eqs. (4) and (6) | `2`, `4` |
| `--apertures` | aperture sizes of the focal-aperture stack | `2 4 ... 16` |
| `--collage_step` | collage shows every n-th depth | `4` |
| `--afi_pixels` | pixels `s,t` to save AFIs for | none |
| `--afi_scale` | nearest-neighbor upsampling of the saved AFIs | `16` |

**Outputs.** Every setting that changes a result is in its file name:

- `mosaic`: `mosaic.png`, `center_view.png`.
- `refocus`: `<shape>_a<size>/refocus_d<d>.png` (`<shape>_afull/` for the largest aperture).
- `allfocus`: `allfocus_s1-<sigma1>_s2-<sigma2>_<stack>.png` and the matching `depth_*.png`/`.npy`.
- `confocal`: `collage_*.png`, `afi_s<s>_t<t>_*.png`, `depth_confocal_*.png`/`.npy`.

## 2. Refocusing an unstructured lightfield: `src/unstructured.py`

```
python src/unstructured.py --video data/part3/chipotle.mov --out results/unstructured/chipotle \
    --step 2 --scale 0.25 --window 140 --template 120 448 60 60
```

Leave out `--template` to pick the rectangle by clicking two opposite corners on the middle frame.

Eq. (9) is computed with `correlate2d` in `valid` mode: the numerator is the window correlated with
the zero-mean template (the `I-bar` term drops out since the zero-mean template sums to 0), and
`I-bar` and the local variance `n * (box(I^2) - I-bar^2)` come from box filters (separable, two
`correlate2d` calls each). The window is centered on the template and clipped to the frame.

| Flag | Meaning | Default |
|---|---|---|
| `--video` | input video | required |
| `--step` | use every n-th frame | `1` |
| `--scale` | resize frames first | `1.0` |
| `--template TOP LEFT WIDTH HEIGHT` | template rectangle in the middle frame | pick it by clicking two opposite corners (prints the matching `--template` values) |
| `--window` | search window side (px) | `200` |

It saves the middle frame (`middle_frame_*.png`, for picking template
coordinates), `refocus_*.png`, `template_*.png` (the
template box on the middle frame) and the per-frame shifts `shifts_*.npy`.

**Capture.** All three videos were filmed with a phone (TODO: model and app) as 10-bit HDR (HLG);
OpenCV decodes them to 8-bit RGB. For `chipotle.mov` the phone was pressed against a window and slid in a
zig-zag; `hamerschlag.mov` was filmed hand-held, also in a zig-zag. TODO: how `chipotle_bars.mov` was moved.

## 3. Capturing a focal-aperture stack (bonus): `src/take_focal_stack.py`

```
python src/take_focal_stack.py --folder squash --ss 1/125
```

Captures a focal-aperture stack with `gphoto2` over USB (tested with a Nikon D3400). It turns off
auto ISO and sets `--iso` and `--quality`, then sweeps the focus once per aperture in `--apertures`.
The shutter speed is scaled by `N^2` from `--ss` (the speed for the first aperture) so `t / N^2` stays
constant, rounded to the closest speed the camera offers. Each sweep runs in one `gphoto2 --shell`
session, since manual focus driving only works in live view: it drives the focus to the end of its
range, then takes `--n_steps` shots, stepping the focus by `--focus_step` between them. Images are saved as
`<folder>/f<N>_<index>.nef` / `.jpg`, with index 0 focused farthest. Set the camera's live-view focus mode
to AF-S first. On macOS the script kills `ptpcamerad` before every `gphoto2` call, since it claims the
camera. At the end it warns about sweeps that missed shots or focus steps.

| Flag | Meaning | Default |
|---|---|---|
| `--folder` | output folder | required |
| `--apertures` | f-numbers to sweep | `1.8 4 9 22` |
| `--n_steps` | focus settings per aperture | `30` |
| `--focus_step` | `manualfocusdrive` step between shots | `150` |
| `--ss` | shutter speed at the first aperture | `1/60` |
| `--iso` | ISO | `400` |
| `--quality` | `raw`, `jpeg` or `both` | `both` |

## 4. Processing the focal-aperture stack (bonus): `src/focal_aperture.py`

```
python src/focal_aperture.py --folder squash --focus_range 5 22 --afi_pixels 50,478 379,712
```

It reads the `f<N>_<index>.nef` captures (linear, via `rawpy`) and divides each by its `t / N^2`
from the EXIF, so every aperture shares one radiometric scale (the exitance of Eq. (7)). It then aligns the magnification change across focus settings with a global scale and shift estimated
by ECC on the smallest aperture. It writes the collage, AFIs, example frames, the depth-from-focus
depth and all-in-focus image (largest aperture), and the confocal stereo depth to
`results/focal_aperture/<folder>/`. Depth is in units of focus setting index. `--skip_load`
reuses the cached aligned stack.

| Flag | Meaning | Default |
|---|---|---|
| `--width` | resize images to this width | `800` |
| `--no_align` | skip the alignment | off |
| `--focus_range FIRST LAST` | only use these focus settings (where something is in focus) | all |
| `--sigma1`, `--sigma2` | Gaussian sigmas of Eqs. (4) and (6) | `2`, `4` |

## Submitted data (`data/`)

The provided lightfield and the raw focal-aperture captures (`squash/`, 3.7 GB) are not included. Images are
JPEG copies of the PNG results, with the result file names, so the settings are in each name.

| Folder | Contents |
|---|---|
| `part1/` | `mosaic.jpg` (sub-aperture mosaic); `refocus/refocus_d<d>.jpg`, the full 41-depth focal stack with the full aperture; `allfocus_*.jpg` and `depth_*.jpg` for sigma1 = 0.5, 1, 2, 4, 8 (sigma2 = 4) and sigma2 = 1, 2, 4, 8, 16 (sigma1 = 2); the focal-aperture `collage_*.jpg`; `afi_s<s>_t<t>_*.jpg`; and the confocal stereo depth `depth_confocal_*.jpg`. |
| `part3/` | The three videos (`chipotle.mov`, `chipotle_bars.mov`, `hamerschlag.mov`) and the refocused images, named `<video>_<tag>.jpg` with the tag of `refocus_<tag>.png`. `chipotle_bars.mov` is re-encoded (HEVC, CRF 24, same resolution and HLG) to fit GitHub's 100 MB limit; the original was 113 MB. |
| `part4/` | Bonus focal-aperture stack (`squash`, focus settings 5-22): example frames `frame_f<N>_<index>_*.jpg`, `collage_*.jpg`, `afi_*.jpg`, the all-in-focus image `allfocus_*.jpg`, and the depth-from-focus `depth_*.jpg` and confocal `depth_confocal_*.jpg` depths. |
