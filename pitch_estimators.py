import numpy as np

def prime_yin_estimator(buffer_size, sample_rate, freq_min, freq_max, thresh=0.1, df_algorithm="tapered_yin"):
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

    # Difference function (sum of squared differences)
    # If we assume our short buffer == kernel and we pre-validate it to a
    # power of 2, we can massively simplify the code.
    def circular_acf(buffer):
        # Regular autocorrelation (ACF), but flipped, raised to min=0, and
        # powered by FFT. Removes all overhead to improve speed (~0.4ms on
        # 32769 after all YIN steps), but partially breaks ACF because
        # point-wise multiplication in FT actually gives circular convolution.
        # Still works surprisingly well for real-time tuning use since where
        # the signal is stationary and noise is low. Additionally, plain ACF
        # is sensitive to tiny imperfections (see the paper), so produces
        # spotty ambient readings (could mitigate with a smoothing filter), a
        # few cents error on simple synthetic data (except with large buffer
        # sizes), and lower robustness to octave errors. The lower precision
        # gets averaged out in real-time use, but still ends up slower than
        # the paper's after considering the buffer size vs accuracy tradeoff.
        sum_sq = np.sum(buffer**2)  # Ends up faster than just conv's max
        fc = np.fft.rfft(buffer)
        conv = np.fft.irfft(fc * np.conj(fc))[:tau_max]
        # Safe to drop all linear terms since CMNDF is insensitive to +y scale
        return sum_sq - conv


    # NOTE: Broken. Works on synthetic data-ish, but doesn't render
    # Could be that scaling just isn't good enough for realistic data,
    # considering the adapted taper ver works fine ish
    def untapered_acf(buffer):
        # The more correct approach to autocorrelation by padding the window,
        # but necessarily loses half the buffer. Seems to end up with worse
        # robustness in some cases, probably because it can't mitigate
        # amplitude variation as well?
        fc1 = np.fft.rfft(buffer)
        fc2 = np.fft.rfft(buffer[:buffer.size // 2], buffer.size)
        conv = np.fft.irfft(fc1 * np.conj(fc2))[:buffer // 2]
        # Lift-to-0 term to too painful, just use max
        return np.max(conv) - conv


    def tapered_acf(buffer):
        # By fixing the circular issues by padding both, we can introduce
        # natural tapering as the window slides off-screen while keeping the
        # full buffer length. May show up to an order of magnitude improvement
        # on synthetic data, but real-time performance is not much different
        # from circular, plus is ~1.5-2x slower on 32768.
        sum_sq = np.sum(buffer**2)
        fc = np.fft.rfft(buffer, buffer.size * 2)
        conv = np.fft.irfft(fc * np.conj(fc))[:tau_max]
        return sum_sq - conv


    def tapered_yin(buffer):
        # Adapted from: https://github.com/patriceguyot/Yin/blob/master/yin.py
        # Similar to eq 6 from the paper. Instead finding the product of each
        # window shift in ACF, which could amplify amplitude defects if the
        # signal is increasing in volume etc, they find the raw difference and
        # square it to achieve the correct scaling. Cumsum is an optimized,
        # O(n) implementation of eq 7's expansion to find every window shift's
        # energy. Finally, this variant pads both sides to add extra tapering,
        # though not in the paper. While ~2x slower than circular ACF at 32768,
        # it does far better from as low as 4096, making it ~4x faster anyways
        # and a tiny 10ms latency. Even a low 4096 performs great on synthetic
        # data (0.00 cents error when noise is low), reduces sporadic ambient
        # readings, more robust to real-world octave errors and background
        # sounds, and can detect C0 comfortably where prior favor 8192.
        x_cumsum = np.concatenate((np.array([0]), (buffer**2).cumsum()))
        fc = np.fft.rfft(buffer, buffer.size * 2)
        conv = np.fft.irfft(fc * np.conj(fc))[:buffer.size]
        return (x_cumsum[buffer.size:0:-1] + x_cumsum[buffer.size] - x_cumsum[:buffer.size] - 2 * conv)[:tau_max]


    def pick_minima(df):
        # Cumulative mean normalized difference function (CMNDF)
        # Adapted from: https://github.com/patriceguyot/Yin/blob/master/yin.py
        # De-prioritizes low-lag correlations, placing ideal solutions near 0
        # Improves over aubio python sample by leveraging np vectorization
        cmndf = np.insert(df[1:] * np.arange(1, df.size) / np.cumsum(df[1:]), 0, 1)
        #cmndf = cmndf + np.linspace(0, 2, num=cmndf.size)  # Forced taper seems to perform worse

        # CMNDF threshold
        # Avoids high-lag mispicks, aka the octave problem. Instead of using
        # min or naively picking the first, we know that good CMNDF troughs
        # will be near-zero. So we pick first within a threshold. Naive
        # iteration is barely faster than vectorized, yet way easier to debug.
        tau = tau_min  # Start scanning from the min period (high-pass)
        while tau < df.size:  # tau_max already applied
            if cmndf[tau] < thresh:
                while tau + 1 < df.size and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                break
            else:
                tau += 1

        if tau >= df.size:  # Shouldn't trigger if thresh was found
            # Last resort: Just pick the global min and pray
            tau = np.argmin(cmndf[tau_min:]) + tau_min

        # Parabolic interpolation
        # By checking adjacent points, we can refine the minima down to
        # subsample resolutions. The original paper undervalues this, but is
        # actually critical for musical use, as half a sample period error may
        # be off by a multiple cents.
        if 0 < tau < df.size - 1:  # Only if adjacent points exist
            # Try to model a tiny parabola and optimize minima
            # Since CMNDF adds a slight lag bias, operate on prior diff_func
            prev_y = df[tau - 1]
            next_y = df[tau + 1]
            denominator = 2 * (prev_y - 2 * df[tau] + next_y)

            if denominator != 0:
                tau = min(max((prev_y - next_y) / denominator + tau, tau_min), df.size - 1)

        return sample_rate / tau

    # Prime lamda by closure
    # Inclusive min, exclusive max stop
    tau_min = int(np.floor(sample_rate / max(freq_max, 1)))
    tau_max = min(int(np.ceil(sample_rate / freq_min)) + 1, buffer_size)


    # Pick sub-strategy
    df_func_map = {
        "tapered_yin": tapered_yin,
        "tapered_acf": tapered_acf,
        "circular_acf": circular_acf,
    }
    df_func = df_func_map[df_algorithm]  # Var to reduce lookup overhead
    # TODO: Probably combine PYIN into this later as another sub-strategy

    # TODO: If viable, might expose functions to user to let them DI into a smaller factory instead of this inner function mess?
    # Can pass max to dfs, and min to minimas

    return lambda buffer: pick_minima(df_func(buffer))
