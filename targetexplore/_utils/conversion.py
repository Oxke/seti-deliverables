import numpy as np
c = 299_792_458

def angle_to_freq(theta, D, c=c):
    theta_FOV = theta / 60 * np.pi/180
    return c / (D * theta_FOV / 1.22) / 1e9
vangle_to_freq = np.vectorize(angle_to_freq)

def freq_to_angle(f, D, c=c):
    theta_FOV = 1.22 * (c/f) / D
    return theta_FOV * 180/np.pi * 60 / 1e9
vfreq_to_angle = np.vectorize(freq_to_angle)
