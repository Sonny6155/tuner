# Plots some f(x) via curses and braille chars

from contextlib import contextmanager
import curses
import time

import numpy as np

# For programmatically resolving 2x4 Unicode Brailles
# Annoyingly non-monotonic, reversed for y height
LEFT_BRAILLE_MAP = [1, 2, 4, 64]
RIGHT_BRAILLE_MAP = [8, 16, 32, 128]
BRAILLE_BLOCK_OFFSET = 10240


@contextmanager
def init_stdscr():
    # A more flexible than provided wrapper, in case of later threading
    # Used via with statement to ensure safe teardown
    try:
        stdscr = curses.initscr()
        curses.noecho()  # Prevent stdin affecting console
        stdscr.keypad(True)  # Consume arrows to prevent weird behaviours
        curses.curs_set(0)  # Hide cursor

        yield stdscr
    finally:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.flushinp()
        curses.endwin()


def render_signal(window, data, roll_i=0):
    """
    Paints f(x) to a curses window. Can accept a rolling window if performance
    is a concern.

    By using braille unicodes for 2x4 subchar resolution, this plots a
    maximum of 2 * window width samples. Data from 0 to 1 (inclusive) are
    rescaled to window height and quantized to 4x res.

    For best results, ensure samples >= maximum window width * 2 and
    preferably an even number. Excess samples will by left-trimmed for use in
    real-time plotting. Any quantized y falling outside of the 0-1 range are
    also dropped.

    :param window: The curses window object to paint to.
    :param data: Circular list/array of y-values between 0-1 inclusive.
    :param roll_i: Current head index. If not a circular list, use 0.
    :raises curses.error: On offscreen write.
    """

    # Though 0 to width-1/height-1 are writable, cursor wrapping may throw an
    # error upon writing to the bottom-right corner. To save some headaches,
    # just discard one col (usually the longest axis).
    window_y, window_x = window.getmaxyx()
    window_x = window_x - 1

    # Grab recent n samples (tail, inclusive of roll_i)
    # Slice x-samples to max braille dot width, trimming excess from left-side
    end_i = roll_i  # Tail
    start_i = end_i - min(len(data), window_x * 2)
    if start_i < 0:
        trimmed = np.concat([data[start_i:], data[:end_i]])
    else:
        trimmed = data[start_i:end_i]

    # Quantize y-data to max braille dot height (stored as char + dot residue)
    # Also flips to curses-space (0,0 is top-left)
    scale_factor = 4 * window_y - 3  # Places 1 exactly on the final dot
    char_y, dot_offsets = np.divmod(np.round(scale_factor * (1 - trimmed)).astype(np.int64), 4)

    # Paint each 2-stride as 2x4 braille char(s)
    for i in range(0, len(char_y) - 2, 2):
        paint_x = i // 2
        paint_y1 = char_y[i]
        paint_y2 = char_y[i+1]

        if paint_y1 == paint_y2:
            if 0 <= paint_y1 < window_y:
                window.addch(paint_y1, paint_x, chr(
                    LEFT_BRAILLE_MAP[dot_offsets[i]] + RIGHT_BRAILLE_MAP[dot_offsets[i+1]] + BRAILLE_BLOCK_OFFSET
                ))
        else:
            if 0 <= paint_y1 < window_y:
                window.addch(paint_y1, paint_x, chr(
                    LEFT_BRAILLE_MAP[dot_offsets[i]] + BRAILLE_BLOCK_OFFSET
                ))
            if 0 <= paint_y2 < window_y:
                window.addch(paint_y2, paint_x, chr(
                    RIGHT_BRAILLE_MAP[dot_offsets[i+1]] + BRAILLE_BLOCK_OFFSET
                ))

    # If sample count is odd for reason, plot last as a standalone char
    if len(char_y) % 2 == 1:
        paint_y = char_y[-1]
        if 0 <= paint_y < window_y:
            window.addch(paint_y, (len(char_y) - 1) // 2, chr(
                LEFT_BRAILLE_MAP[dot_offsets[0]] + BRAILLE_BLOCK_OFFSET
            ))
        

if __name__ == "__main__":
    # Quick test to ensure it works
    rolling_display = np.zeros(200)
    roll_i = 0
    data_length = 800

    # Sine with period of 80 dots (40 chars) 
    #data_source = np.sin(2 * np.pi * np.arange(data_length) / 80) / 2 + 0.5

    # Mixed sines, with some out-of-bounds
    data_source = (np.sin(2 * np.pi * np.arange(data_length) / 80) + 0.5 * np.sin(2 * np.pi * np.arange(data_length) / 40)) / 2 + 0.5

    # More complex example
    data_source = (
        0.4 * np.sin(2 * np.pi * np.arange(data_length) / 160) + \
        0.3 * np.sin(2 * np.pi * np.arange(data_length) / 80) + \
        0.5 * np.sin(2 * np.pi * np.arange(data_length) / 60) + \
        0.4 * np.sin(2 * np.pi * np.arange(data_length) / 30)
    ) / 2 + 0.5

    with init_stdscr() as stdscr:
        for y in data_source:
            try:
                stdscr.erase()

                # Prepare data for render, appending to circular list
                rolling_display[roll_i] = y
                roll_i = roll_i + 1 if roll_i < len(rolling_display) - 1 else 0
        
                render_signal(stdscr, rolling_display, roll_i=roll_i)
                stdscr.refresh()
            except curses.error:
                # Attempt to warn user of temporary window overflow
                try:
                    stdscr.addstr(0, 0, "Frame dropped:\nResizing or window too small")
                    stdscr.refresh()
                except curses.error:
                    pass

            time.sleep(0.016)
