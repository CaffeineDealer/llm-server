import json
import re
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--results-file", default="results_raw.json")
parser.add_argument("--output-csv", default="parsed_results.csv")
args = parser.parse_args()

with open(args.results_file, "r") as f:
    results_raw = json.load(f)

# Metric name -> regex pattern to extract it
metrics_patterns = {
    "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
    "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
}

parsed_rows = []

for entry in results_raw:
    row = {
        "num_prompts": entry["num_prompts"],
        "input_len": entry["input_len"],
        "output_len": entry["output_len"],
        "returncode": entry["returncode"],
        "avg_power_watts": entry["avg_power_watts"],
        "duration_s": float(entry["duration_s"])
    }

    total_output_tokens = int(entry["num_prompts"]) * int(entry["output_len"])
    if row["avg_power_watts"] is not None and row["duration_s"] and total_output_tokens > 0:
        energy_joules = row["avg_power_watts"] * row["duration_s"]
        row["energy_per_token_j"] = energy_joules / total_output_tokens
    else:
        row["energy_per_token_j"] = None

    # Skip parsing if this combo timed out or failed (no stdout to parse)
    if entry["stdout"] is None:
        for metric_name in metrics_patterns:
            row[metric_name] = None
        parsed_rows.append(row)
        continue

    # Try to extract each metric from this entry's stdout text
    for metric_name, pattern in metrics_patterns.items():
        match = re.search(pattern, entry["stdout"])
        row[metric_name] = float(match.group(1)) if match else None

    parsed_rows.append(row)

# Write everything to a CSV file for easy viewing/plotting
fieldnames = list(parsed_rows[0].keys())
with open(args.output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(parsed_rows)

print(f"Parsed {len(parsed_rows)} rows, saved to {args.output_csv}")