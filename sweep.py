import subprocess
import itertools
import json
import threading
import time


# Helper function to sample GPU power
def sample_gpu_power(stop_event, samples):
    while not stop_event.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        try:
            watts = float(result.stdout.strip())
            samples.append(watts)
        except ValueError:
            pass
        time.sleep(0.2)

MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"

# Param grid
num_prompts_list = [1, 10, 50, 100]
input_len_list = [128, 512, 1024]
output_len_list = [128, 256]

# Output
results_raw = []

for num_prompts, input_len, output_len in itertools.product(
    num_prompts_list, input_len_list, output_len_list
):
    print(f"Running: Number of Prompts={num_prompts}, Input Prompt Length={input_len}, Output Prompt Length={output_len}")

    power_samples = []
    stop_event = threading.Event()

    sampler_thread = threading.Thread(target=sample_gpu_power, args=(stop_event, power_samples))
    sampler_thread.start()

    # Build terminal command as a list of strings for subprocess
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--model", MODEL,
        "--dataset-name", "random",
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--request-rate", "inf",
        "--num-prompts", str(num_prompts),
    ]

    start_time = time.time()
    # Run the command in the terminal 
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1000)
        stdout_text = result.stdout
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        # If a run hangs past 300s, skip it and log it instead of killing the whole sweep
        print(f"  TIMEOUT — skipping this combo")
        stdout_text = None
        returncode = "timeout"
    finally:
        stop_event.set()
        sampler_thread.join()    
    duration_s = time.time() - start_time

    avg_power_watts = sum(power_samples) / len(power_samples) if power_samples else None

    # save it
    results_raw.append({
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "stdout": stdout_text,
        "returncode": returncode,
        "avg_power_watts": avg_power_watts,
        "duration_s": duration_s,
    })

with open("results_raw.json", "w") as f:
    json.dump(results_raw, f, indent=2)

print(f"Done. Collected {len(results_raw)} results.")
