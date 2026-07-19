# tuner

A graphical precision tuner and general-purpose pitch tracker.

## Features
- Real-time, TUI pitch plotting.
- Tuning mode for ease of use.
- Advanced controls for power users:
    - Input bandpass and displayed range.
    - Estimator buffer and block stream sizing.
    - Algorithm selection (WIP, limited to YIN variants).

## Quick Start
### Requirements
Recommended to install via pip, using miniconda (or similar) to isolate your Python environment. Check `requirements.txt` for additional replication notes.
```bash
pip install -r requirements.txt
```

*Conda historically had issues with PortAudio and Python versioning. If that's resolved, feel free to use that instead, but note the different package name.*

### Examples
General-purpose pitch tracking:
```bash
python tuner.py
```

Fine-tune a single note (to +/-1 semitone, accepts Hz or SPN):
```bash
python tuner.py -f 440
python tuner.py -f A4
```

Use a non-default microphone:
```bash
python tuner.py --list-devices
python tuner.py -d "Some Mic Name"
python tuner.py -d 2
```

For more options:
```bash
python tuner.py -h
python tuner.py --list-tips
```

*Tested for MacOS. Should work fine on Linux and Windows, but check the `sounddevice` docs for additional components. Linux may require system package `portaudio` (or`libportaudio2` on Ubuntu), while Windows may want some code tweaks to use ASIO for sub-10ms latencies.*

## Technical Details
### Current Algorithms
- YIN: Autocorrelation-based, F0 detector. One of the simpler options, but fast (with FFT) and effective.
    - Well-suited for tuning in low-noise environments on sustained notes.
    - May underperform for instruments with particularly loud overtones and for overly non-stationary signals. Functional for vocals, but very poor for regular speech.
    - Works well on default 4096 buffer size (~100 ms latency). Increasing to 8192 may slightly improve stability and low frequency range.
    - Other variants included are experimental. For best performance, go with the default tapered YIN.

### Limitations
- Pitch is currently estimated only from a short audio buffer. While this makes sense for YIN, some algorithms have their own way of reading real-time data or take into account the entire audio history. Unfortunately, those algorithms will simply have to make do with this short buffer approach even if it means a worse estimator, or include their own state management.
- Buffer/block sizes are limited to powers of 2. This keeps things fast and easy to work with, as it completely avoids the trouble of padding/trimming for FFT and other windowing logic. But otherwise, it's a completely arbitrary constraint.

### Project Layout
```text
README.md
demo/                # Experiments, comparisons, etc.
pitch_estimators.py  # Pitch algorithms.
render.py            # Helpers for curses and braille drawing.
requirements.txt
tuner.py             # Entry point. Handles args and init.
```

## Motivation
This originally started as a simple CLI guitar tuner, made around 2026-01-07. Physical tuners are annoying to use or bandlimit too aggressively. Tuning websites are generally a lot easier to use, but often lack precise sub-cent indicators and tend to be a privacy nightmare. So the goal was to write a simple script so that I could tune from any computer, uses as few dependencies as possible, and runs only locally.

After some initial success, more features were added to make it both more of a general purpose pitch tracking tool, and as my permanent replacement to all other tuners.

It turned out that the prototype was already good enough to outperform the web tuner I was previously using in precision and stability. With some adjustments to the algorithm and its parameters, it became relatively robust to overdrive/distortion and light noise, which meant I could tune without resetting all my amp settings everytime. It was also stable enough for use with generic instruments and vocals outside of a tuning context. However, the raw outputs were sometimes still too noisy for the original HIGH/LOW + cent deviation readings. A point-in-time reading was also too hard to use for non-stationary signals (and guitar strings jump/drift a lot within the first 3s of a pluck). By implementing a graphical pitch-over-time display, I could visualize and characterize pitch movement purely by eye, pick a suitable tuning, then simply adjust down to a sub-cent precision. Further args were later added to assist visual precision and further mitigate misclassification of overtones common to some instruments.

In addition to all of above, this project also doubles as a personal research side-project. F0 pitch detection seems to be a rather deep rabbithole. I would like to use this repo as a testing ground to tinker with re-implement more algorithms by hand (at some point). While some may improve performance on select instruments, most will exist for comparison's sake.

## Future Work
Only if I get around to it.

- Simple smoothing filters and generic (if not algorithm-specific) voicing flags should help denoise tracking outputs. This should help a ton with YIN, whose outputs get sporadic in low signal-to-noise environments.
- The UX still sucks. CLI init args are cumbersome. The two most viable options here are direct TUI pitch input, or simpler CLI aliases (e.g. guitar strings 1-6). The framework for either already exist.
- The helper info text is a bit bloated. This really needs to be split up into some sort of `--list <subject>` option...
- I would like to explore other real-time algorithms, or at least hack one into real-time use. Issue is, every single one is a pain to implement. Or require some bloated library written in C/Rust then ported to Python. Or are NN-based.
