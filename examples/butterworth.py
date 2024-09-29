"""
Example provided at: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
"""
from scipy.signal import butter, lfilter, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=5, output="ba"):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output=output)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, output="sos"):
    butter_values = butter_bandpass(lowcut, highcut, fs, order=order, output=output)
    if output == "ba":
        b, a = butter_values
        y = lfilter(b, a, data)
    elif output == "sos":
        sos = butter_values
        y = sosfilt(sos, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 600.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [2, 4, 6]: #[3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, output="ba")
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.05  # sinewave amplitude (used here to emphasize our desired frequency)
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += 0.01 * np.cos(2 * np.pi * 510 * t + 0.1)  # another frequency in the lowcut and highcut range
    x += 0.01 * np.cos(2 * np.pi * 520 * t + 0.1)  # another frequency in the lowcut and highcut range
    x += 0.01 * np.cos(2 * np.pi * 530 * t + 0.1)  # another frequency in the lowcut and highcut range
    x += 0.01 * np.cos(2 * np.pi * 540 * t + 0.1)  # another frequency in the lowcut and highcut range
    x += 0.01 * np.cos(2 * np.pi * 550 * t + 0.1)  # another frequency in the lowcut and highcut range
    x += 0.01 * np.cos(2 * np.pi * 1200 * t + 0.1) # another frequency in the lowcut and highcut range
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal')
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()

if __name__ == "__main__":
    run()