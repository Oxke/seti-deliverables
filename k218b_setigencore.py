#!/usr/bin/env python
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from inject_signal import inject_signal

DATA_FOLDER = "../random-data/"
OUTPUT_FOLDER = "../k218b/"
WINDOW_SIZE = 1024 # size in freq bins of injected signal

seticore_env = os.environ.copy()
seticore_env["HDF5_PLUGIN_PATH"] = "/home/obs/.conda/envs/seticore/lib/python3.12/site-packages/hdf5plugin/plugins"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
all_files = glob.glob(os.path.join(DATA_FOLDER, "k2-18b*/*0.fil"))

# --- Define processing function for one file ---
def process_file(filepath):
    print(f"processing {filepath}")
    if already_processed(filepath): return "Already processed, skipping"
    try:
        print("  injecting signal") # injected describes the injected signal
        clean_path, injected_path, injected = inject_signal(filepath, OUTPUT_FOLDER)

        print("  running seticore")
        subprocess.run(["seticore", clean_path, "-m", "-5.2", "-M", "5.2",
                        "--output", out_clean := (clean_path[:-3] + ".dat")],
                       check=True, env=seticore_env, stdout=subprocess.DEVNULL)
        subprocess.run(["seticore", injected_path, "-m", "-5.2", "-M", "5.2",
                        "--output", out_mod := (injected_path[:-3] + ".dat")],
                       check=True, env=seticore_env, stdout=subprocess.DEVNULL)

        print("  computing diff")
        diff_process = subprocess.run(["diff", out_clean, out_mod], capture_output=True, text=True)
        diff_output = diff_process.stdout
        line_count = len(diff_output.strip().splitlines())

        print("  saving diff")
        base_name = os.path.basename(filepath).replace(".fil", "")
        diff_filename = f"diff_{line_count}_lines_{base_name}.txt"
        diff_path = os.path.join(OUTPUT_FOLDER, diff_filename)
        inject_signal_theory = os.path.join(OUTPUT_FOLDER,
                                            f"{base_name}.expected.txt")
        with open(diff_path, "w") as f:
            f.write(diff_output)
        with open(inject_signal_theory, "w") as f:
            f.write(injected)

        return f"Processed: {filepath}"
    except Exception as e:
        return f"Failed: {filepath} with error {e}"

def already_processed(filepath):
    base_name = os.path.basename(filepath).replace(".fil", "")
    inject_signal_theory = os.path.join(OUTPUT_FOLDER,
                                        f"{base_name}.expected.txt")
    return os.path.exists(inject_signal_theory)


if __name__ == "__main__":
    batch_size = 7
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process_file, f) for f in batch]
            for future in as_completed(futures):
                print(future.result())
