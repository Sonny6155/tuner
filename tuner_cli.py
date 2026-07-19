import pitch_estimators

import argparse

import numpy as np
import sounddevice as sd

# Stick with lower-level Stream API instead of play/rec/playrec, since it's
# closer to PortAudio/PyAudio's style and more powerful. Mixing them will
# cause duplicate Streams.

C0 = (2**-4.75) * 440  # Equal temperament, scaled around A4=440
PITCH_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_info(freq):
    # Returns nearest scientific pitch notation (e.g. 440 -> "A4") name and its cent deviation
    # First, rescale into the number of semitones from C0
    semitone = 12 * np.log2(freq / C0)
    rounded_semitone = round(semitone)
    
    # Find notation of nearest semitone, and the cent difference
    letter = PITCH_SCALE[int(rounded_semitone) % 12]
    spn = f"{letter}{rounded_semitone // 12}"
    cent_diff = (semitone - rounded_semitone) * 100

    return spn, cent_diff


def tune(stream, buffer, block_size, estimator, lines=20):
    # TODO: can make this an async callback with mutex later on, allowing fixed fps reads on separate thread
    # leave as blocking for now

    # Set up lines for rendering
    line_buffer = [""] * lines
    print("\n" * lines, end="")

    while True:
        # Write chunk to rolling buffer
        # Don't use np.roll, which tends to be unnecessarily slow. Concat
        # seems faster than manual slice+broadcast on larger buffers.
        arr, overflowed = stream.read(block_size)
        buffer = np.concat([buffer[:-block_size], arr.reshape(-1)])

        # Compute current pitch and output SPN and cent
        # TODO: this should be an in place overwrite, or curses line write
        # Overflow are super rare with sane block settings, so just ignore atm
        spn, cent = pitch_info(estimator(buffer))

        # TODO: Employ smoothing and accept as arg?
        # One Euro Filter for lightweight jitter smoothing that still allows
        # low-latency changes when it matters. Should also slightly smooth any
        # sporadic estimation spikes.
        # try minimum cutoff 1, cutoff slope 0.007

        # Rolling line display (via escape codes)
        if cent == 0:
            indicator = "Good"
        elif cent > 0:
            indicator = "High"
        else:
            indicator = "Low "

        line_buffer.pop(0)
        line_buffer.append(f"{indicator} | {spn}, {cent:.2f}")
        print("\033[A\033[2K\r" * lines + "\n".join(line_buffer))
        
        # TODO: Move to curses. Also make a frequency analyzer TUI and place on separate pane


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detects the fundamental frequency (f0) from a mic."
    )
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="List available input devices, then exit.",
    )
    parser.add_argument(
        "-d", "--device", default=sd.default.device[0],
        help="The input device to use, by ID or substring. PortAudio tries to pick one by default if omitted.",
    )
    parser.add_argument(
        "-b", "--buffer-size", type=int, default=8192,
        help="Number of samples to estimate with each tick. Must be a power of 2. Ideal value depends on the algorithm and lowest accepted pitch. For untapered YIN using FFT, 4096 comfortably detects most notes, while 16384 or higher may be required for <20Hz (like C0). Lower may simply fail. (default: %(default)s)",
    )
    parser.add_argument(
        "-B", "--block-size", type=int, default=2048,
        help="Number of samples to read from input at a time. Must be less than or equal to buffer size. Note that a block size of 1024 runs at ~43FPS (~23ms) for a 44100Hz sample rate, and that few terminals even render much faster. Overly miniscule blocks or weak hardware may also skip data (unless block size = buffer size). (default: %(default)s)",
    )
    # TODO: might add async pitch estimation of buffer. block size still holds for data update rate.
    parser.add_argument(
        "-a", "--algorithm", default="yin", choices=["yin", "tapered_yin"],  # TODO: Add more later
        help="The f0 pitch estimation algorithm to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-f", "--freq-range", type=int, nargs=2, default=[20, 20000],
        help="Min and max frequencies in Hz to limit to, if known and allowed by algorithm. For guitar tuning, try using 20 to 800 (default: %(default)s)",
    )
    args = parser.parse_args()

    # Validate args
    if args.buffer_size < 1 or (args.buffer_size & (args.buffer_size - 1)) != 0:
        # We want a power of 2 for pad-less FFT usage
        parser.error("Buffer size must be a positive power of 2.")
    elif args.buffer_size < args.block_size:
        parser.error(f"Buffer size must be larger than block size ({args.block_size}).")
    elif args.freq_range[0] > args.freq_range[1]:
        parser.error("Max freq cannot be less that min freq.")
    elif args.device is None:
        parser.error("No device found. Do you have a microphone attached/built-in?")

    if args.list_devices:
        print(sd.query_devices(kind="input"))
    else:
        # Enter tune mode
        try:
            # 20kHz is accepted for (sane) mic sample rates (per Nyquist),
            # while 20Hz is possible with a >=4096 buffer size (per SR / BS).
            # That said, each strategy employ their own heuristics that may
            # make them perform far better (or worse) with different settings.
            # For example, harmonic comb methods fail beyond (SR/4)Hz.
            buffer = np.zeros(args.buffer_size)
            sample_rate = sd.query_devices(args.device)["default_samplerate"]

            algorithm_map = {
                "yin": pitch_estimators.prime_yin_estimator(
                    args.buffer_size,
                    sample_rate,
                    *args.freq_range,
                    taper=False,
                ),
                "tapered_yin": pitch_estimators.prime_yin_estimator(
                    args.buffer_size,
                    sample_rate,
                    *args.freq_range,
                    taper=True,
                ),
            }

            with sd.InputStream(
                device=args.device,
                blocksize=args.block_size,
                samplerate=sample_rate,
                channels=1,  # TODO: Check what higher channels actually does
                # Leave dithering on, suppressing artificial harmonics etc
            ) as stream:
                # Raw stream read Python buffers to skip an np copy
                tune(stream, buffer, args.block_size, algorithm_map[args.algorithm])

        except KeyboardInterrupt:
            parser.exit("\nInterrupted by user. Exiting...")

