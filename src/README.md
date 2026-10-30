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
python src/unstructured.py --video data/part3/chipotle.mov --out results/unstructured/chipotle --step 2 --scale 0.25 \
    --size 60 --window 140 --template 120 448 --template 330 375 --template 410 570 --template 270 800
python src/unstructured.py --video data/part3/chipotle_bars.mov --out results/unstructured/chipotle_bars --step 3 --scale 0.25 \
    --size 60 --window 140 --template 110 450 --template 325 380 --template 395 610
python src/unstructured.py --video data/part3/hamerschlag.mov --out results/unstructured/hamerschlag \
    --step 2 --scale 0.5 --size 60 --window 260 --template 170 515 --template 100 588 --template 110 90
```

Eq. (9) is computed with `correlate2d` in `valid` mode: the numerator is the window correlated with
the zero-mean template (the `I-bar` term drops out since the zero-mean template sums to 0), and
`I-bar` and the local variance `n * (box(I^2) - I-bar^2)` come from box filters (separable, two
`correlate2d` calls each). The window is centered on the template and clipped to the frame.

| Flag | Meaning | Default |
|---|---|---|
| `--video` | input video | required |
| `--step` | use every n-th frame | `1` |
| `--scale` | resize frames first | `1.0` |
| `--template ROW COL` | top-left corner of a template in the middle frame; repeat for several focus points | none |
| `--size` | template side (px) | `60` |
| `--window` | search window side (px) | `200` |

It saves the middle frame (`middle_frame_*.png`, for picking template
coordinates) and, for each template, `refocus_*.png`, `template_*.png` (the
template box on the middle frame) and the per-frame shifts `shifts_*.npy`.

TODO: how the video was captured (device, settings, motion).

## 3. Capturing a focal-aperture stack (bonus): `src/take_focal_stack.py`

TODO

## 4. Processing the focal-aperture stack (bonus): `src/focal_aperture.py`

```
python src/focal_aperture.py --folder squash --afi_pixels 200,300 300,500
```

It reads the `f<N>_<index>.nef` captures (linear, via `rawpy`) and divides each by its `t / N^2`
from the EXIF. Brightness still jumps by up to ~35% between shots, so it scales each shot to the
same mean over the central half of the frame, then corrects the remaining per-pixel exitance
differences (vignetting) between apertures,
and aligns the magnification change across focus settings with a global scale and shift estimated
by ECC on the smallest aperture. It writes the collage, AFIs, example frames, the depth-from-focus
depth and all-in-focus image (largest aperture), and the confocal stereo depth to
`results/focal_aperture/<folder>/`. Depth is in units of focus setting index. `--skip_load`
reuses the cached aligned stack.

| Flag | Meaning | Default |
|---|---|---|
| `--width` | resize images to this width | `800` |
| `--exitance_sigma` | blur of the per-pixel exitance estimate (`0`: one gain per aperture) | `40` |
| `--no_align` | skip the alignment | off |
| `--focus_range FIRST LAST` | only use these focus settings (where something is in focus) | all |
| `--sigma1`, `--sigma2` | Gaussian sigmas of Eqs. (4) and (6) | `2`, `4` |
| `--confocal_sigma` | blur of the confocal cost before the argmin | `2` |

## Submitted data (`data/`)

TODO

| Folder | Contents |
|---|---|
| `part1/` | TODO |
| `part3/` | The three videos (`chipotle.mov`, `chipotle_bars.mov`, `hamerschlag.mov`) and the refocused images, named `<video>_<tag>.jpg` with the tag of `refocus_<tag>.png`. `chipotle_bars.mov` is re-encoded (HEVC, CRF 24, same resolution and HLG) to fit GitHub's 100 MB limit; the original was 113 MB. |
