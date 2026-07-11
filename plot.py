import csv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input-csv", default="parsed_results.csv")
args = parser.parse_args()

rows = []
with open(args.input_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

groups = {}
for row in rows:
    key = (row["input_len"], row["output_len"])
    groups.setdefault(key, []).append(row)

plt.style.use("dark_background")
palette = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
unique_combos = sorted(groups.keys(), key=lambda k: (int(k[0]), int(k[1])))
colors = {combo: palette[i % len(palette)] for i, combo in enumerate(unique_combos)}

# num_prompts values, derived from the data instead of hardcoded
x_labels = sorted(set(int(r["num_prompts"]) for r in rows))
x_positions = range(len(x_labels))


def plot_metric(metric_key, ylabel, title, output_png):
    plt.figure(figsize=(8, 6))
    for (input_len, output_len), group_rows in groups.items():
        group_rows.sort(key=lambda r: int(r["num_prompts"]))
        y = [float(r[metric_key]) if r[metric_key] else None for r in group_rows]

        label = f"in={input_len}, out={output_len}"
        plt.plot(x_positions, y, marker="o", markersize=6, linewidth=2,
                  color=colors[(input_len, output_len)], label=label)

    plt.xticks(x_positions, x_labels)
    plt.xlabel("Concurrency (num_prompts)", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"results/{output_png}", dpi=150)
    plt.close()
    print(f"Saved results/{output_png}")


plot_metric("output_token_throughput", "Output tokens/sec", "Output Token Throughput vs Concurrency", "output_throughput.png")
plot_metric("request_throughput", "requests/sec", "Request Throughput vs Concurrency", "request_throughput.png")
plot_metric("mean_ttft_ms", "Mean TTFT (msec)", "Time to First Token vs Concurrency", "ttft.png")
plot_metric("mean_tpot_ms", "Mean TPOT (msec)", "Time per Output Token vs Concurrency", "tpot.png")
plot_metric("avg_power_watts", "Avg GPU Power (W)", "GPU Power Draw vs Concurrency", "gpu_power.png")
plot_metric("energy_per_token_j", "Energy per token (J)", "Energy Efficiency vs Concurrency", "energy_per_token.png")