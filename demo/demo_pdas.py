import pitch_estimators

import timeit

import numpy as np


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


if __name__ == "__main__":
    sample_rate = 44100
    min_freq = 1  # Since we are testing very low freqs
    max_freq = 20000
    buffer_sizes = [
        1024,
        4096,
        8192,
        16384,
        32768,
    ]
    algorithm_map = {
        "tapered_yin": lambda n: pitch_estimators.prime_yin_estimator(
            n,
            sample_rate,
            min_freq,
            max_freq,
            df_algorithm="tapered_yin",
        ),
        "tapered_acf_yin": lambda n: pitch_estimators.prime_yin_estimator(
            n,
            sample_rate,
            min_freq,
            max_freq,
            df_algorithm="tapered_acf",
        ),
        "circular_acf_yin": lambda n: pitch_estimators.prime_yin_estimator(
            n,
            sample_rate,
            min_freq,
            max_freq,
            df_algorithm="circular_acf",
        ),
    }
    signal_map = {
        "sineC0": lambda n: np.sin(2 * np.pi * C0 / sample_rate * np.arange(n)),
        "sineA4": lambda n: np.sin(2 * np.pi * 440 / sample_rate * np.arange(n)),
        "sineC4": lambda n: np.sin(2 * np.pi * 261.6256 / sample_rate * np.arange(n)),
        "sineF9": lambda n: np.sin(2 * np.pi * 11175.30 / sample_rate * np.arange(n)),
        "A4Harmonics": lambda n: np.sin(2 * np.pi * 440 / sample_rate * np.arange(n)) + 0.3 * np.cos(2 * np.pi * 880 / sample_rate * np.arange(n)) + 0.7 * np.cos(2 * np.pi * 1760 / sample_rate * np.arange(n)),
        "A4HarmonicsWeakF0": lambda n: 0.3 * np.sin(2 * np.pi * 440 / sample_rate * np.arange(n)) + 0.7 * np.sin(2 * np.pi * 880 / sample_rate * np.arange(n)) + np.sin(2 * np.pi * 1760 / sample_rate * np.arange(n)),
        "A4WeakSubharmonics": lambda n: np.sin(2 * np.pi * 440 / sample_rate * np.arange(n)) + 0.1 * np.sin(2 * np.pi * 220 / sample_rate * np.arange(n)),
        "C4Mixed": lambda n: np.sin(2 * np.pi * 261.6256 / sample_rate * np.arange(n)) + 0.2 * np.cos(2 * np.pi * 347 / sample_rate * np.arange(n)),  # Seems to be become quite unstable above 0.3
        "C4GaussianNoise": lambda n: np.sin(2 * np.pi * 261.6256 / sample_rate * np.arange(n)) + 0.5 * np.random.uniform(0, 1, size=n),
    }
    repeats = 200 #400

    for buffer_size in buffer_sizes:
        for algo_name, algo in algorithm_map.items():
            for signal_name, signal in signal_map.items():
                estimator = algo(buffer_size)
                buffer = signal(buffer_size)
                timing = timeit.timeit(
                    "estimator(buffer)",
                    globals=globals(),
                    number=repeats,
                )
                freq = estimator(buffer)
                spn, cent_diff = pitch_info(freq)
                print(f"{buffer_size} {algo_name} {signal_name}, Time (ms): {timing * 1000 / repeats:.5f}, f0: {freq:.2f} {spn} {cent_diff:.02f}")
