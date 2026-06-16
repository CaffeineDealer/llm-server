# %% Essential libraries
import torch
import numpy as np

import matplotlib.pyplot as plt

from transformers import Qwen3VLForConditionalGeneration
from transformers import AutoTokenizer

# %% Set up the LLM model on GPU
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct-FP8",
    device_map="cuda"
)

# %%
def make_gain_hook(gain):
    def hook(module, input, output):
        # output is the attention output tensor
        # scale it by gain factor
        if isinstance(output, tuple):
            return (output[0] * gain,) + output[1:]
        return output * gain
    return hook

# %%
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-32B-Instruct-FP8")
inputs = tokenizer("The first person to summit K2 was", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}


outputs_baseline = model.generate(
    **inputs,
    max_new_tokens=10,
    output_scores=True,
    return_dict_in_generate=True,
    do_sample=False
    )

gfactor = 3 # gain value
target_layer = 10 
hook = model.model.language_model.layers[target_layer].self_attn.register_forward_hook(make_gain_hook(gain=gfactor))

# re-register hook
outputs_boosted = model.generate(
    **inputs,
    max_new_tokens=10,
    output_scores=True,
    return_dict_in_generate=True,
    do_sample=False
    )

print("BASELINE:", tokenizer.decode(outputs_baseline.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
print("BOOSTED:", tokenizer.decode(outputs_boosted.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

hook.remove()  # disable hook

scores_baseline = outputs_baseline.scores[0]
scores_boosted = outputs_boosted.scores[0]

probs_baseline = torch.softmax(scores_baseline, dim=-1)
probs_boosted = torch.softmax(scores_boosted, dim=-1)

top10_baseline = torch.topk(probs_baseline, k=10, dim=-1)
top10_boosted = torch.topk(probs_boosted, k=10, dim=-1)

# Printing token probabilites
# print("BASELINE top 10:")
# for token_id, prob in zip(top10_baseline.indices[0], top10_baseline.values[0]):
#     print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")

# print("BOOSTED top 10:")
# for token_id, prob in zip(top10_boosted.indices[0], top10_boosted.values[0]):
#     print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")

# for step, (scores_b, scores_boost) in enumerate(zip(outputs_baseline.scores, outputs_boosted.scores)):
#     probs_b = torch.softmax(scores_b, dim=-1)
#     probs_boost = torch.softmax(scores_boost, dim=-1)
#     top10_b = torch.topk(probs_b, k=10, dim=-1)
#     top10_boost = torch.topk(probs_boost, k=10, dim=-1)
    
#     print(f"\n--- Token {step+1} ---")
#     print("BASELINE:")
#     for token_id, prob in zip(top10_b.indices[0], top10_b.values[0]):
#         print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")
#     print("BOOSTED:")
#     for token_id, prob in zip(top10_boost.indices[0], top10_boost.values[0]):
#         print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")

# %%

n_steps = len(outputs_baseline.scores)
k = 10

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, outputs, title in zip(axes, [outputs_baseline, outputs_boosted], ["BASELINE", "BOOSTED"]):
    grid = np.zeros((k, n_steps))
    x_labels = []
    cell_labels = []

    for step in range(n_steps):
        probs = torch.softmax(outputs.scores[step], dim=-1)
        top10 = torch.topk(probs, k=k, dim=-1)
        grid[:, step] = top10.values[0].cpu().numpy()
        cell_labels.append([tokenizer.decode(t).strip() for t in top10.indices[0]])
        picked = tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1] + step]).strip()
        x_labels.append(picked)

    im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"rank {i+1}" for i in range(k)])
    ax.set_title(title)

    for step in range(n_steps):
        for rank in range(k):
            ax.text(step, rank, cell_labels[step][rank], ha='center', va='center', fontsize=6, color='white')

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
# %%
