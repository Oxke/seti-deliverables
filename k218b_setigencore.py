#!/usr/bin/env python
import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from inject_signal import inject_signal

parser = argparse.ArgumentParser(
    description="Process filterbank files by adding some signal"
    + "and trying to retrieve it"
)
parser.add_argument(
    "--nproc", "-N", default=7, help="Number of synchronous processes", type=int
)
parser.add_argument(
    "--input", "-i", default="../random-data/", help="Input folder", nargs="+"
)
parser.add_argument(
    "--filter", "-x", default="k2-18b*/*K2-18b*0.fil", help="Input file format mask"
)
parser.add_argument("--output", "-o", default="../k218b/", help="Output folder")
parser.add_argument(
    "-dr",
    default=[-5, 5, 200],
    help="Drift rates, 1 is drift rate, 2 is range, 200 points, 3 is range and number of points, so `-dr -5 5 200` is 200 drift rates from -5 to 5",
    nargs="+",
    type=float,
)
parser.add_argument(
    "--log10_snr",
    "-snr",
    default=[1, 3, 200],
    help="Like drift rates, 1 to 3 arguments, but it's the log10 of the drift rate",
    nargs="+",
    type=float,
)
parser.add_argument(
    "--snr_threshold", "-t", default=10, help="SNR threshold for seticore", type=float
)
parser.add_argument(
    "--n_files", "-f", default=0, help="Number of files to analyze", type=int
)
args = parser.parse_args()

N_FILES = args.n_files
N_PROC = args.nproc
SNR_THRESH = args.snr_threshold
DATA_FOLDER = args.input
OUTPUT_FOLDER = args.output
DR = args.dr
N_DR = 200
if len(DR) == 1:
    DR = MIN_DR = MAX_DR = DR[0]
if len(DR) >= 2:
    MIN_DR = DR[0]
    MAX_DR = DR[1]
    if len(DR) == 3:
        N_DR = DR[2]
    if len(DR) >= 4:
        parser.error("You must provide 1 to 3 dr floats, not more")
    DR = None

SNR = args.dr
N_SNR = 200
if len(SNR) == 1:
    SNR = MIN_SNR = MAX_SNR = SNR[0]
if len(SNR) >= 2:
    MIN_SNR = SNR[0]
    MAX_SNR = SNR[1]
    if len(SNR) == 3:
        N_SNR = SNR[2]
    if len(SNR) >= 4:
        parser.error("You must provide 1 to 3 dr floats, not more")
    SNR = None

seticore_env = os.environ.copy()
seticore_env["HDF5_PLUGIN_PATH"] = (
    "/home/obs/.conda/envs/seticore/lib/python3.12/site-packages/hdf5plugin/plugins"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
all_files = glob.glob(os.path.join(DATA_FOLDER, args.filter))


# --- Define processing function for one file ---
def process_file(filepath):
    print(f"processing {filepath}")
    if already_processed(filepath):
        return "Already processed"
    try:
        print("  injecting signal")  # injected describes the injected signal
        clean_path, injected_path, injected = inject_signal(
            filepath,
            OUTPUT_FOLDER,
            min_dr=MIN_DR,
            max_dr=MAX_DR,
            n_dr=N_DR,
            dr=DR,
            snr=SNR,
            min_snr=MIN_SNR,
            max_snr=MAX_SNR,
            n_snr=N_SNR,
        )
        print("  running seticore")
        print(str(SNR_THRESH))
        subprocess.run(
            [
                "seticore",
                clean_path,
                "-m",
                str(MIN_DR - 0.2),
                "-M",
                str(MAX_DR + 0.2),
                "-s",
                str(SNR_THRESH),
                "--output",
                out_clean := (clean_path[:-3] + ".dat"),
            ],
            check=True,
            env=seticore_env,
        )
        subprocess.run(
            [
                "seticore",
                injected_path,
                "-m",
                str(MIN_DR - 0.2),
                "-M",
                str(MAX_DR + 0.2),
                "-s",
                str(SNR_THRESH),
                "--output",
                out_mod := (injected_path[:-3] + ".dat"),
            ],
            check=True,
            env=seticore_env,
        )

        print("  computing diff")
        diff_process = subprocess.run(
            ["diff", out_clean, out_mod], capture_output=True, text=True
        )
        diff_output = diff_process.stdout
        line_count = len(diff_output.strip().splitlines())

        print("  saving diff")
        base_name = os.path.basename(filepath).replace(".fil", "")
        diff_filename = f"diff_{line_count}_lines_{base_name}.txt"
        diff_path = os.path.join(OUTPUT_FOLDER, diff_filename)
        inject_signal_theory = os.path.join(OUTPUT_FOLDER, f"{base_name}.expected.txt")
        with open(diff_path, "w") as f:
            f.write(diff_output)
        with open(inject_signal_theory, "w") as f:
            f.write(injected)

        return f"Processed: {filepath}"
    except Exception as e:
        return f"Failed: {filepath} with error {e}"


def already_processed(filepath):
    base_name = os.path.basename(filepath).replace(".fil", "")
    inject_signal_theory = os.path.join(OUTPUT_FOLDER, f"{base_name}.expected.txt")
    return os.path.exists(inject_signal_theory)


if __name__ == "__main__":
    batch_size = N_PROC
    if N_FILES > 0:
        all_files = all_files[:N_FILES]
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process_file, f) for f in batch]
            for future in as_completed(futures):
                print(future.result())
