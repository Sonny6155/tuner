import numpy as np

def prime_yin_estimator(buffer_size, sample_rate, freq_min, freq_max, thresh=0.1, taper=False):
    # Prime by closure
    w = buffer_size

    # We can effectively bandpass by ignoring low/high periods during calcs
    # Assumes freq_min <= freq_max
    if freq_max > 0:
        # Try to fit in window, but min must be 1+ to avoid later div by 0s
        tau_min = max(min(int(np.floor(sample_rate / freq_max)), w - 1), 1)
    else:
        tau_min = w 

    if freq_min > 0:
        # Exclusive stop
        tau_max = min(int(np.ceil(sample_rate / freq_min)) + 1, w)
    else:
        tau_max = w

    # With the right setup, both untapered and tapered are blazingly fast at
    # ~0.1ms for unwindowed 8192 buffers and still ~0.5ms for 32768. FFT beats
    # naive from as low as 1024.
    
    # In terms of accuracy, the untapered seems to be behave a bit better.
    # Higher buffer sizes can help accuracy somewhat (and works poorly below
    # 2048). While parabolic interpolation seemed only weakly effective on
    # synthetic sample, it did see a large drop in gross error on sizes 16k+.
    # In real-time usage, however, the subsampled resolution reduced the worst
    # resolution errors from as much as 10 cents down to less than 1 for
    # guitar base pitches. It also greatly improved the detection rate of a
    # highly distorted, finger-plucked E2 (especially on higher buffer sizes),
    # where it previously saw B3 unless plucked gently at 12th fret. However,
    # increasing buffer size also seemed to increase jitter (by eye) from
    # sub-cent to ~3 centers at 16/32k, so would benefit from a smoothing
    # for front-end display.

    # Note that YIN struggles on real-world signals at 2048 or less buffer
    # sizes, and barely fails C0 at 4096. This can be mitigated to an extent
    # by greatly increasing the threshold, but is generally not a great idea.
    # Best to just use 8192 or more.

    def yin_estimator(buffer):
        # Difference function (sum of squared differences)
        # Loosely adapted from:
        # - https://github.com/patriceguyot/Yin/blob/master/yin.py
        # - https://stackoverflow.com/questions/973370/fast-average-square-difference-function
        # If we assume our short buffer == kernel and we pre-validate it to a
        # power of 2, we can massively simplify the code. We won't use tau_max
        # as the window to skip over FFT padding concerns. Finally, we can
        # drop all linear terms, so long as shape remains and global min = 0.
        # That also means we must use squaring to match IFFT, not just abs.
        sum_sq = np.sum(buffer**2)
        fc = np.fft.rfft(buffer)  # Sticking to reals skips tons of calcs
        conv = np.fft.irfft(fc * fc.conjugate(), w)[:tau_max]
        # Apparently trimming outside irfft is way faster if not factor of 2?
        diff_func = sum_sq - conv

        # Cumulative mean normalized difference function (CMNDF)
        # Adapted from: https://github.com/patriceguyot/Yin/blob/master/yin.py
        # De-prioritizes low-lag correlations, placing ideal solutions near 0
        # Improves over aubio python sample by leveraging np vectorization
        cmndf = np.insert(diff_func[1:] * np.arange(1, tau_max) / np.cumsum(diff_func[1:]), 0, 1)
        #cmndf = cmndf + np.linspace(0, 2, num=cmndf.size)  # Forced taper seems to perform worse

        # CMNDF threshold
        # Avoids high-lag mispicks, aka the octave problem. Instead of using
        # min or naively picking the first, we know that good CMNDF troughs
        # will be near-zero. So we pick first within a threshold. Naive
        # iteration is barely faster than vectorized, yet way easier to debug.
        tau = tau_min  # Start scanning from the min period (high-pass)
        while tau < tau_max:
            if cmndf[tau] < thresh:
                while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                break
            else:
                tau += 1

        if tau >= tau_max:  # Shouldn't trigger if thresh was found
            # Last resort: Just pick the global min and pray
            tau = np.argmin(cmndf[tau_min:]) + tau_min

        # Parabolic interpolation
        # By checking adjacent points, we can refine the minima down to
        # subsample resolutions. Thought the original paper undervalues this
        # this is actually critical for this is critical for musical use, as
        # Half a sample period error may be off by a multiple cents. Works
        # best with 16k or more buffer size.
        if 0 < tau < tau_max - 1:  # Only if adjacent points exist
            # Try to model a tiny parabola and optimize minima
            # Since CMNDF adds a slight lag bias, operate on prior diff_func
            prev_y = diff_func[tau - 1]
            next_y = diff_func[tau + 1]
            denominator = 2 * (prev_y - 2 * diff_func[tau] + next_y)

            if denominator != 0:
                tau = min(max((prev_y - next_y) / denominator + tau, tau_min), tau_max - 1)

        return sample_rate / tau  # Should be validated not 0

    
    def tapered_yin_estimator(buffer):
        # This version uses the original paper's tapered eq 6, which the ref
        # Github code adapts. Can still skip windowing and linear scalings.
        # Alright in practice, but failed miserably on synthetic data. Also
        # several times slower.
        x_cumsum = np.concatenate((np.array([0]), (buffer**2).cumsum()))
        fc = np.fft.rfft(buffer, w * 2)  # x2 pad is important in this ver
        conv = np.fft.irfft(fc * fc.conjugate())[:w]  # Must outer not inner trim
        diff_func = (x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv)[:tau_max]

        cmndf = np.insert(diff_func[1:] * np.arange(1, tau_max) / np.cumsum(diff_func[1:]), 0, 1)

        tau = tau_min
        while tau < tau_max:
            if cmndf[tau] < thresh:
                while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
            tau += 1

        if tau >= tau_max:
            tau = np.argmin(cmndf[tau_min:]) + tau_min

        if 0 < tau < tau_max - 1:
            prev_y = diff_func[tau - 1]
            next_y = diff_func[tau + 1]
            denominator = 2 * (prev_y - 2 * diff_func[tau] + next_y)

            if denominator != 0:
                tau = min(max((prev_y - next_y) / denominator + tau, tau_min), tau_max - 1)

        return sample_rate / tau


    # Pick sub-strategy
    return tapered_yin_estimator if taper else yin_estimator
