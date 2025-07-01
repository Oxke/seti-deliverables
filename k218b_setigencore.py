import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from inject_signal import inject_signal

DATA_FOLDER = "../../random-data/"
OUTPUT_FOLDER = "../../k218b/"
WINDOW_SIZE = 1024 # size in freq bins of injected signal
SETICORE = "HDF5_PLUGIN_PATH=/home/obs/.conda/envs/seticore/lib/python3.12/site-packages/hdf5plugin/plugins seticore"

os.makedirs(output_dir, exist_ok=True)

# --- Collect all files matching *0.fil ---
all_files = glob.glob(os.path.join(dir1, "k2-18b*/*0.fil"))

# --- Define processing function for one file ---
def process_file(filepath):
    print(f"processing {filepath}")
    try:
        print("  injecting signal")
        clean_path, injected_path = inject_signal(filepath, OUTPUT_FOLDER)

        print("  running seticore")
        subprocess.run([SETICORE, clean_path,
                        "--output", out_clean := (clean_path[:-3] + ".dat")],
                       check=True)
        subprocess.run([SETICORE, clean_path,
                        "--output", out_mod := (injected_path[:-3] + ".dat")],
                       check=True)

        print("  computing diff")
        diff_process = subprocess.run(["diff", out_clean, out_mod], capture_output=True, text=True)
        diff_output = diff_process.stdout
        line_count = len(diff_output.strip().splitlines())

        print("  saving diff")
        base_name = os.path.basename(filepath).replace(".fil", "")
        diff_filename = f"diff_{line_count}_lines_{base_name}.txt"
        diff_path = os.path.join(OUTPUT_FOLDER, diff_filename)
        with open(diff_path, "w") as f:
            f.write(diff_output)
            
        return f"Processed: {filepath}"
    except Exception as e:
        return f"Failed: {filepath} with error {e}"

# --- Process files in batches of 10 ---
batch_size = 10
for i in range(0, len(all_files), batch_size):
    batch = all_files[i:i+batch_size]
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(process_file, f) for f in batch]
        for future in as_completed(futures):
            print(future.result())
