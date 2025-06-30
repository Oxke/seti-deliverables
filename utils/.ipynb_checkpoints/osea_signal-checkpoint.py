#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

def osea_path(f_start, drift_rate, amplitude, omega, mu):
    def path(t):
        return amplitude * np.sqrt(np.abs(np.sin(omega * t))) * np.exp(-mu * t)
    return lambda t: f_start + drift_rate * t + path(t)

if __name__ == "__main__":
    T = np.linspace(0, 300, 301)
    p = osea_path(40, 1, 100, 0.07, 0.01)
    plt.plot(T, p(T))
    plt.show()
