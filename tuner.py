import pitch_estimators
from render import init_stdscr, render_signal

import argparse
import curses

import numpy as np
import sounddevice as sd

# Stick with lower-level Stream API instead of play/rec/playrec, since it's
# closer to PortAudio/PyAudio's style and more powerful. Mixing them will
# cause duplicate Streams.

C0 = (2**-4.75) * 440  # Equal temperament, scaled around A4=440
PITCH_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SEMITONE_RATIO = 2**(1/12)  # Not to be relied upon for too many multiplies


def freq_to_pitch(freq):
    # Returns nearest scientific pitch notation (e.g. 440 -> "A4") name and its cent deviation
    # First, rescale into the number of semitones from C0
    semitone = 12 * np.log2(freq / C0)
    rounded_semitone = round(semitone)
    
    # Find notation of nearest semitone, and the cent difference
    letter = PITCH_SCALE[int(rounded_semitone) % 12]
    spn = f"{letter}{rounded_semitone // 12}"
    cent_diff = (semitone - rounded_semitone) * 100

    return spn, cent_diff


def pitch_to_freq(spn, cent_diff):
    num_start = -2 if "-" in spn else -1
    semitone = PITCH_SCALE.index(spn[:num_start]) + int(spn[num_start:]) * 12 + cent_diff / 100
    return 2**(semitone/12) * C0


def tune(stream, buffer_size, block_size, estimator, display_min, display_max, centerline=False):
    # Set up rolling buffers for estimator input and live freq output
    buffer = np.zeros(args.buffer_size)  # Rolls by copy for read speed
    freq_display = np.zeros(args.buffer_size)  # Rolls by circular index
    freq_roll_i = 0
    
    # Cache freq-to-display rescaling constants
    # Using (log2(freq) - log2(min)) / (log2(max) - log2(min)), where min is
    # pre-validated
    rescaling_min = np.log2(display_min)
    rescaling_linear = np.log2(display_max) - rescaling_min

    # Launch curses screen
    with init_stdscr() as stdscr:
        curses.start_color()

        while True:
            # Update rolling buffer and estimate on whole thing
            # Concat outspeeds np.roll and manual slice on larger buffers
            arr, overflowed = stream.read(block_size)
            buffer = np.concat([buffer[:-block_size], arr.reshape(-1)])
            freq = estimator(buffer)

            try:
                stdscr.erase()
                window_y, window_x = stdscr.getmaxyx()
                window_x -= 1
                # TODO: Move to new multiple newwin setup later, freq_window and stats_window

                # Underlay center indicator
                if centerline:
                    stdscr.addstr(window_y // 2 - 1, 0, "-" * (window_x - 1), curses.A_DIM)

                # Update rolling freq display, rescaled to pitch and within 0 to 1
                freq_display[freq_roll_i] = (np.log2(freq) - rescaling_min) / rescaling_linear
                freq_roll_i = freq_roll_i + 1 if freq_roll_i < len(freq_display) - 1 else 0
                render_signal(stdscr, freq_display, roll_i=freq_roll_i)

                # TODO: Employ smoothing and accept as arg?
                # One Euro Filter for lightweight jitter smoothing that still allows
                # low-latency changes when it matters. Should also slightly smooth any
                # sporadic estimation spikes.
                # try minimum cutoff 1, cutoff slope 0.007

                # Output SPN and cent as overlay
                spn, cent = freq_to_pitch(freq)
                # Rolling line display (via escape codes)
                if cent > 2:
                    indicator = "High"
                elif cent < -2:
                    indicator = "Low"
                else:
                    indicator = "Good"
                stdscr.addstr(window_y-1, 0, f"{indicator} | {spn}, {cent:.2f}")

                stdscr.refresh()
            except curses.error:
                # Attempt to warn user of temporary window overflow
                try:
                    stdscr.addstr(0, 0, "Frame dropped:\nResizing or window too small")
                    stdscr.refresh()
                except curses.error:
                    pass


def freq_check(x):
    # A useful wrapper to validate a float freq or it convert from SPN
    if isinstance(x, float) or isinstance(x, int):
        return x
    else:
        try:
            return float(x)
        except ValueError:
            return pitch_to_freq(x, 0)  # May raise ValueError or IndexError


def list_tips():
    # Helper to tuning tips
    return """
Instruments:
- Guitar: [E2, A2, D3, G3, B3, E4]. Filtering detection between 20 and 800
  also works.
- Piano: 88-key ranges A0 to B8, but unlikely to ever be in equal temperament
  due to tuning concerns around the inharmonics of stiff strings and nearby
  strings. Loud and rich inharmonics make notes extremely hard to detect.
- Vocals: Singing may range as wide as E2-A5 (or more), while useful overtones
  go much higher. For general speech detection, consider a cepstral-like
  algorithm and using the full 20kHz input range. Autocorrelation may suffice
  for clear singing.

Algorithms:
- YIN: Autocorrelation-based. Some robustness against partials or uncorrelated
  noise, but otherwise fairs poorly against loud harmonics and real-world
  background voices/noise. Good choice for F0 detection and instrument tuning
  in quiet environments.
    - Tapered YIN: Highly recommended. Squared difference backed by FFT
      autocorrelation, modified from the paper to taper. Much better accuracy
      and at small buffer sizes than autocorrelation, making it both fast to
      compute and very low latency. Also more robust to octave errors and
      non-stationary than ACF. Recommended to use default 4096 buffer size.
    - Tapered ACF: Plain autocorrelation (ACF) but tapers due to the FFT
      windowing used. More robust than other ACF methods against subharmonics,
      but ACF still inherently magnifies certain amplitude errors. Produces a
      bit more sporadic readings at ambient and a tad lower precision for live
      usage (thicker line of jitter). Recommended to use 8192 buffer size, but
      4096 may work fine.
    - Circular ACF: Removes all overhead for speed at a similar buffer sizes,
      but uses circular ACF as a result. Comparable to tapered ACF in real-world
      usage, though slightly less robust to background sounds and harmonics.
      Definitely worse on synthetic data. 8192 buffer size is the minimum to
      achieve C0.
- Cepstral: TODO
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detects the fundamental frequency (f0) from a mic."
    )

    # Core options
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available input devices, then exit.",
    )
    parser.add_argument(
        "--list-tips", action="store_true",
        help="List common instrument tunings as SPNs, then exit.",
    )
    parser.add_argument(
        "-f", "--fine-tune", type=freq_check,
        help="Pitch to tune for (shorthand for advanced options). If given, limits display to +/-1 semitone, detector input range to +/- 3 semitones, and adds a centerline tuning indicator. Works best on larger windows. Accepts Hz or SPN (e.g. -f 80, -f E2).",
    )

    # Advanced options
    parser.add_argument(
        "--input-range", type=freq_check, nargs=2, default=[C0, 20000],
        help="Min-max pitches to limit the algorithm to. Behavior is algorithm-dependent, but generally emulates bandpass preprocessing. May improve noise resistance, or worsen detection accuracy (especially for cepstral-type algorithms). Accepts Hz or SPN (e.g. --display_range 12.34 F#6). (default: [\"C0\", 20000])",
    )
    parser.add_argument(
        "--display-range", type=freq_check, nargs=2,
        help="Min-max pitches to render. Accepts Hz or SPN (e.g. --display_range 12.34 F#6). If not specified and a display_target was not set, it will default to equal input_range.",
    )
    parser.add_argument(
        "--centerline", action="store_true",
        help="Enables an approximate centerline indicator to assist in tuning. Works best on larger windows.",
    )
    parser.add_argument(
        "-b", "--buffer-size", type=int, default=4096,
        help="Number of samples to estimate with each tick. Must be a power of 2. May affect the accuracy or lowest pitch possible for some algorithms, but also increases latency. (default: %(default)s)",
    )
    parser.add_argument(
        "-B", "--block-size", type=int, default=2048,
        help="Number of samples to read from input at a time. Must be less than or equal to buffer size. Note that a block size of 1024 runs at ~43FPS (~23ms) for a 44100Hz sample rate, and that few terminals even render much faster. Overly tiny blocks or weak hardware may also skip data (unless block size = buffer size). (default: %(default)s)",
    )
    parser.add_argument(
        "-a", "--algorithm", default="tapered_yin", choices=["tapered_yin", "tapered_acf_yin", "circular_acf_yin"],  # TODO: Add more later
        help="The f0 pitch estimation algorithm to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--device", default=sd.default.device[0],
        help="The input device to use, by ID or substring. PortAudio tries to pick one by default if omitted.",
    )
    args = parser.parse_args()

    # Validate args
    if args.buffer_size < 1 or (args.buffer_size & (args.buffer_size - 1)) != 0:
        # We want a power of 2 for pad-less FFT usage
        parser.error("Buffer size must be a positive power of 2.")
    elif args.buffer_size < args.block_size:
        parser.error(f"Buffer size must be larger than block size ({args.block_size}).")
    elif args.device is None:
        parser.error("No device found. Do you have a microphone attached or built-in?")

    if args.fine_tune:
        # Override input/display ranges
        args.input_range = [
            args.fine_tune / (SEMITONE_RATIO**3),
            args.fine_tune * (SEMITONE_RATIO**3),
        ]
        args.display_range = [
            args.fine_tune / SEMITONE_RATIO,
            args.fine_tune * SEMITONE_RATIO,
        ]
        args.centerline = True
    else:
        # Validate/set freq args
        if not 1 <= args.input_range[0] < args.input_range[1]:
            # A 1s freq is mostly nonsensical. Min=1 saves some headaches.
            parser.error("Input freq range must be 1 <= min < max.")
        elif args.display_range is None:
            # Inherit unspecified display range from bandpass
            args.display_range = args.input_range.copy()
        elif not 1 <= args.display_range[0] < args.display_range[1]:
            # Due to log scaling, we need some frequency to start doubling from
            parser.error("Display freq range must be 1 <= min < max.")

    # Start main code
    if args.list_devices:
        print(sd.query_devices(kind="input"))
    elif args.list_tips:
        print(list_tips())
    else:
        # Enter tune mode
        try:
            sample_rate = sd.query_devices(args.device)["default_samplerate"]

            algorithm_map = {
                "tapered_yin": pitch_estimators.prime_yin_estimator(
                    args.buffer_size,
                    sample_rate,
                    *args.input_range,
                    df_algorithm="tapered_yin",
                ),  # Thresh doesn't affect much, and default seems stable
                "tapered_acf_yin": pitch_estimators.prime_yin_estimator(
                    args.buffer_size,
                    sample_rate,
                    *args.input_range,
                    df_algorithm="tapered_acf",
                ),
                "circular_acf_yin": pitch_estimators.prime_yin_estimator(
                    args.buffer_size,
                    sample_rate,
                    *args.input_range,
                    df_algorithm="circular_acf",
                ),  # Thresh doesn't affect much, and default seems stable
            }

            with sd.InputStream(
                device=args.device,
                blocksize=args.block_size,
                samplerate=sample_rate,
                channels=1,
                # Leave dithering on, suppressing artificial harmonics etc
            ) as stream:
                # Raw stream read Python buffers to skip an np copy
                tune(
                    stream,
                    args.buffer_size,
                    args.block_size,
                    algorithm_map[args.algorithm],
                    *args.display_range,
                    args.centerline,
                )

        except KeyboardInterrupt:
            parser.exit("\nInterrupted by user. Exiting...")

