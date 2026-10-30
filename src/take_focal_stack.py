import subprocess
import sys
from pathlib import Path
import argparse
import math
import tempfile

QUALITIES = {
    "raw": "NEF (Raw)",
    "jpeg": "JPEG Fine",
    "both": "NEF+Fine",
}
# manualfocusdrive=1000 repeats used to reach the end of the focus range; extra ones just error out
EDGE_DRIVES = 10


def release_camera():
    """Stop the macOS camera daemon, which claims the camera and makes gphoto2 fail with
    'Could not claim the USB device'. launchd restarts it, so do this before every call."""
    if sys.platform == "darwin":
        subprocess.run(["pkill", "-9", "-x", "ptpcamerad"], stderr=subprocess.DEVNULL)


def gphoto(*args):
    """Run a gphoto2 command and fail if it returns an error."""
    release_camera()
    subprocess.run(
        ["gphoto2", *args],
        check=True
    )


def current_config(name):
    """Current value of a camera config, e.g. 'AF-S' for focusmode2."""
    release_camera()
    out = subprocess.run(
        ["gphoto2", "--get-config", name],
        capture_output=True,
        text=True,
        check=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("Current:"):
            return line.split(" ", 1)[1].strip()
    raise RuntimeError(f"gphoto2 did not report a current value for {name}")


def config_choices(name):
    """All values a camera config accepts, e.g. ['f/3.5', 'f/4', ...] for f-number."""
    release_camera()
    out = subprocess.run(
        ["gphoto2", "--get-config", name],
        capture_output=True,
        text=True,
        check=True
    ).stdout
    return [line.split(" ", 2)[2].strip() for line in out.splitlines() if line.startswith("Choice:")]


def parse_number(value):
    """Numeric value of an f-number or shutter speed string ('f/3.5', '1/60', '30', '0.5s'),
    or None for non-numeric choices such as 'Bulb'."""
    value = value.removeprefix("f/").removesuffix("s")
    try:
        if "/" in value:
            num, den = value.split("/")
            return float(num) / float(den)
        return float(value)
    except ValueError:
        return None


def closest_shutter_speed(seconds, choices):
    """Camera shutter speed closest to `seconds` in stops."""
    timed = [c for c in choices if parse_number(c)]
    return min(timed, key=lambda c: abs(math.log2(parse_number(c) / seconds)))


def shell_commands(n_steps, focus_step):
    """gphoto2 --shell commands for one focal sweep at the current aperture.
    Manual focus driving needs the viewfinder on, which only lasts for one gphoto2 session,
    so the whole sweep goes through a single --shell process (see the hints)."""
    commands = ["set-config /main/actions/viewfinder=1"]
    # Drive to the end of the focus range so every aperture sweeps the same focus settings.
    # Past the end the camera errors with "manual focus stepping too small", which the shell ignores.
    commands += ["set-config /main/actions/manualfocusdrive=1000"] * EDGE_DRIVES
    for i in range(n_steps):
        commands.append("capture-image-and-download")
        if i < n_steps - 1:
            # The camera can drop out of live view after a capture, and a focus drive outside it
            # fails ("works only in LiveView mode"), skipping a focus step
            commands.append("set-config /main/actions/viewfinder=1")
            commands.append(f"set-config /main/actions/manualfocusdrive={-focus_step}")
    # Leave live view so the next aperture is applied before its sweep starts
    commands.append("set-config /main/actions/viewfinder=0")
    return commands


def gphoto_shell(commands, filename):
    """Run commands in one `gphoto2 --shell` session, downloading captures to `filename`
    (a gphoto2 pattern), and return its output lines. --force-overwrite keeps gphoto2 from
    prompting on stdin, which would swallow the next command."""
    release_camera()
    proc = subprocess.Popen(
        ["gphoto2", "--force-overwrite", "--filename", str(filename), "--shell"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    proc.stdin.write("\n".join(commands + ["exit"]) + "\n")
    proc.stdin.close()
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return lines


def download_leftovers(output_dir):
    """Download files still pending on the camera, e.g. from an interrupted run. Otherwise the
    camera answers the edge drives with 'I/O in progress' and the sweep's first download is stale."""
    gphoto("--force-overwrite", "--wait-event-and-download=2s",
           "--filename", str(output_dir / "leftover_%f.%C"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture a focal-aperture stack with gphoto2.")
    parser.add_argument("--folder", required=True, help="Set output folder")
    parser.add_argument("--apertures", nargs="+", default=["1.8", "4", "9", "22"],
                        help="f-numbers to sweep; include the smallest and largest of the lens")
    parser.add_argument("--n_steps", type=int, default=30, help="Focus settings per aperture")
    parser.add_argument("--focus_step", type=int, default=150, help="manualfocusdrive step between shots")
    parser.add_argument("--ss", default="1/60", help="Shutter speed")
    parser.add_argument("--iso", default="400", help="ISO")
    parser.add_argument("--quality", default="both", choices=list(QUALITIES), help="Save RAW, JPEG or both")
    args = parser.parse_args(argv)

    output_dir = Path(args.folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    gphoto("--set-config", f"/main/capturesettings/imagequality={QUALITIES[args.quality]}")
    gphoto("--set-config", "/main/imgsettings/autoiso=Off")
    gphoto("--set-config", f"/main/imgsettings/iso={args.iso}")

    focus_mode = current_config("/main/capturesettings/focusmode2")
    if focus_mode != "AF-S":
        sys.exit(f"Live view focus mode is {focus_mode}; set it to AF-S on the camera "
                 "so manualfocusdrive works.")

    f_numbers = {parse_number(c): c for c in config_choices("/main/capturesettings/f-number")}
    shutter_speeds = config_choices("/main/capturesettings/shutterspeed")
    base_time = parse_number(args.ss)
    base_aperture = float(args.apertures[0])
    warnings = []

    for aperture in args.apertures:
        if float(aperture) not in f_numbers:
            sys.exit(f"f/{aperture} is not available on this lens; choices: {list(f_numbers.values())}")
        # Exposure goes as t / N^2, so scale --ss (given for the first aperture) to keep it constant.
        # Shutter speeds come in 1/3 stops, so the EXIF ExposureTime is what to normalize by.
        ss = closest_shutter_speed(base_time * (float(aperture) / base_aperture) ** 2, shutter_speeds)
        print(f"f/{aperture}: shutter speed {ss}")
        gphoto("--set-config", f"/main/capturesettings/f-number={f_numbers[float(aperture)]}")
        gphoto("--set-config", f"/main/capturesettings/shutterspeed={ss}")
        download_leftovers(output_dir)
        sweep_dir = Path(tempfile.mkdtemp(dir=output_dir))
        log = gphoto_shell(shell_commands(args.n_steps, args.focus_step), sweep_dir / "%03n.%C")

        # Number each format by shot, since %n counts files and a NEF+Fine shot saves two
        for ext in {file.suffix for file in sweep_dir.iterdir()}:
            files = sorted(sweep_dir.glob(f"*{ext}"), key=lambda f: int(f.stem))
            if len(files) != args.n_steps:
                warnings.append(f"f/{aperture}: {len(files)} {ext} files for {args.n_steps} shots")
            for i, file in enumerate(files):
                file.rename(output_dir / f"f{aperture}_{i:03d}{file.suffix}")
        sweep_dir.rmdir()

        # A failed drive leaves this aperture's focus settings out of line with the others
        if not any("stepping too small" in line for line in log):
            warnings.append(f"f/{aperture}: never reached the end of the focus range")
        failed = sum(f"value {-args.focus_step} for configuration entry" in line for line in log)
        if failed:
            warnings.append(f"f/{aperture}: {failed} of {args.n_steps - 1} focus steps failed")

    for warning in warnings:
        print("WARNING:", warning)
    print("Done.")


if __name__ == "__main__":
    main()
