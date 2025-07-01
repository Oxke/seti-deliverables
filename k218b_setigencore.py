import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from inject_signal import inject_signal

DATA_FOLDER = "../../random-data/"
OUTPUT_FOLDER = "../../k218b/"
WINDOW_SIZE = 1024 # size in freq bins of injected signal

os.makedirs(output_dir, exist_ok=True)

# --- Collect all files matching *0.fil ---
all_files = glob.glob(os.path.join(dir1, "k2-18b*/*0.fil"))

# --- Define processing function for one file ---
def process_file(filepath):
    try:
        clean_path, injected_path = inject_signal(filepath, OUTPUT_FOLDER)

        subprocess.run(["HDF5_PLUGIN_PATH seticore", clean_path, "--output", clean_path]check=True)

        with open(command_out2, "w") as fout2:
            subprocess.run(["my_command", output2], stdout=fout2, check=True)

        # Compute diff and count lines
        diff_process = subprocess.run(["diff", command_out1, command_out2], capture_output=True, text=True)
        diff_output = diff_process.stdout
        line_count = len(diff_output.strip().splitlines())

        # Save diff
        base_name = os.path.basename(filepath).replace(".fil", "")
        diff_filename = f"{base_name}_diff_{line_count}_lines.txt"
        diff_path = os.path.join(output_dir, diff_filename)
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
