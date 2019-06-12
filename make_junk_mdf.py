import asammdf
import numpy as np
import sys


def generate_sine_signal(name, freq, amplitude=1, length=10, sample_rate=0.01, noise=False):
    t = np.linspace(0, length, np.ceil(length/sample_rate)+1)
    y = np.sin(2 * np.pi * freq * t)*amplitude
    if noise:
        y = np.add(y, np.random.normal(0, .2*amplitude, len(y)))
    return asammdf.Signal(y, t, name=name)


if __name__ == '__main__':
    sigs = [generate_sine_signal('signal_1', freq=2, amplitude=2, length=100, sample_rate=0.01, noise=True),
            generate_sine_signal('signal_2', freq=1, amplitude=1, length=30, sample_rate=0.01, noise=False),
            generate_sine_signal('signal_3', freq=50, amplitude=.5, length=120, sample_rate=0.0001, noise=True),
            generate_sine_signal('signal_4', freq=.5, amplitude=5, length=60, sample_rate=0.001, noise=True),
            generate_sine_signal('signal_5', freq=2, amplitude=5, length=60, sample_rate=0.01, noise=False),
            generate_sine_signal('signal_6', freq=1, amplitude=3, length=75, sample_rate=0.001, noise=True),
            generate_sine_signal('signal_7', freq=10, amplitude=7, length=100, sample_rate=0.001, noise=True),
            generate_sine_signal('signal_8', freq=7, amplitude=7, length=100, sample_rate=0.001, noise=True),
            generate_sine_signal('signal_9', freq=7, amplitude=1, length=100, sample_rate=0.01, noise=False),
            generate_sine_signal('signal_10', freq=2, amplitude=2, length=100, sample_rate=0.01, noise=True)]
    mdf = asammdf.MDF()
    mdf.append(sigs)
    mdf.save('./sample_data.mf4')