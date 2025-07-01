import setigen as stg
import blimpy as bl
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from copy import deepcopy as dc
from os import path

OUTPUT_FOLDER = "../../deliverables/injected/"
WINDOW_SIZE = 1024 # size in freq bins of injected signal

def make_frame_s(frame, min_freq_index, dr, snr, l, r, imgsave_path=None, output_folder=OUTPUT_FOLDER):
    frame_s = frame.get_slice(l,r)
    signal = frame_s.add_signal(
        stg.constant_path(f_start=frame_s.get_frequency(index=min_freq_index),
                          drift_rate=dr),
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=100*u.Hz),
        stg.constant_bp_profile(level=1)
    )
    if imgsave_path:
        frame_s.plot()
        plt.savefig(imgsave_path)
    return frame_s

def inject_signal(fil_filepath, output_folder=OUTPUT_FOLDER, **kwargs):
    frame = stg.Frame(waterfall=fil_filepath)
    if "l" in kwargs:
        l = kwargs["l"]
    else:
        L = np.linspace(0, frame.data.shape[1] - WINDOW_SIZE, 50, dtype=int)
        l = np.random.choice(L)

    if "snr" in kwargs:
        if kwargs["snr"] is None:
            SNR = 10 ** np.linspace(kwargs["min_snr"],
                                    kwargs["max_snr"],
                                    kwargs["n_snr"])
            snr = np.random.choice(SNR)
        else:
            snr = kwargs["snr"]
    else:
        SNR = 10 ** np.linspace(1, 3, 200)
        snr = np.random.choice(SNR)

    # I don't need it for now so I won't implement it, but if dr is passed then
    # it's a bit different since I have to calculate mfi

    DR = np.linspace(-5, 5, 100)
    MFI = (WINDOW_SIZE*frame.df - frame.data.shape[0]*frame.dt*DR) / (2 * frame.df)
    index_dr = np.random.randint(DR.size)
    mfi = np.round(MFI[index_dr]) ; dr = DR[index_dr]

    r = l + WINDOW_SIZE

    if "output_name" in kwargs:
        if (output_name:=kwargs["output_name"]).endswith(".h5"):
            output_name = output_name[:-3]
    else:
        output_name = path.splitext(path.basename(fil_filepath))[0]

    frame.save_h5(cleanpath := output_folder + output_name + ".clean.h5")

    image_path = output_folder + output_name + ".signal.png"
    frame.data[:, l:r] = make_frame_s(frame, mfi, dr, snr, l, r, imgsave_path=image_path, output_folder=output_folder).data
    frame.save_h5(modpath := output_folder + output_name + ".mod.h5")

    return cleanpath, modpath, f"{mfi = }\n{dr = }\n{snr = }\nwindow = {l} - {r}"

if __name__ == "__main__":
    filepath = "../../random-data/guppi_60703_16858_008036_TIC286923464_ON_0001.0000.fil"
    print(inject_signal(filepath))
