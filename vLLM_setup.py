
# %%
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct-FP8",
    device_map="cuda"
)

def make_gain_hook(gain):
    def hook(module, input, output):
        # output is the attention output tensor
        # scale it by gain factor
        if isinstance(output, tuple):
            return (output[0] * gain,) + output[1:]
        return output * gain
    return hook

# %%
from transformers import AutoTokenizer

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
#print("BASELINE:", tokenizer.decode(outputs_baseline[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
print("BASELINE:", tokenizer.decode(outputs_baseline.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

hook = model.model.language_model.layers[10].self_attn.register_forward_hook(make_gain_hook(gain=0.5))

# re-register hook
outputs_boosted = model.generate(
    **inputs,
    max_new_tokens=10,
    output_scores=True,
    return_dict_in_generate=True,
    do_sample=False
    )
#print("BOOSTED:", tokenizer.decode(outputs_boosted[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
print("BOOSTED:", tokenizer.decode(outputs_boosted.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

hook.remove()  # disable hook

import torch
scores_baseline = outputs_baseline.scores[0]
scores_boosted = outputs_boosted.scores[0]

probs_baseline = torch.softmax(scores_baseline, dim=-1)
probs_boosted = torch.softmax(scores_boosted, dim=-1)

top10_baseline = torch.topk(probs_baseline, k=10, dim=-1)
top10_boosted = torch.topk(probs_boosted, k=10, dim=-1)


# print("BASELINE top 10:")
# for token_id, prob in zip(top10_baseline.indices[0], top10_baseline.values[0]):
#     print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")

# print("BOOSTED top 10:")
# for token_id, prob in zip(top10_boosted.indices[0], top10_boosted.values[0]):
#     print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")

for step, (scores_b, scores_boost) in enumerate(zip(outputs_baseline.scores, outputs_boosted.scores)):
    probs_b = torch.softmax(scores_b, dim=-1)
    probs_boost = torch.softmax(scores_boost, dim=-1)
    top10_b = torch.topk(probs_b, k=10, dim=-1)
    top10_boost = torch.topk(probs_boost, k=10, dim=-1)
    
    print(f"\n--- Token {step+1} ---")
    print("BASELINE:")
    for token_id, prob in zip(top10_b.indices[0], top10_b.values[0]):
        print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")
    print("BOOSTED:")
    for token_id, prob in zip(top10_boost.indices[0], top10_boost.values[0]):
        print(f"  {tokenizer.decode(token_id)}: {prob.item():.4f}")

# %%
hook = model.model.language_model.layers[32].self_attn.register_forward_hook(make_gain_hook(gain=2))
# %%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-32B-Instruct-FP8")
inputs = tokenizer("The capital of France is", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))# %%


# %%
for name, param in model.named_parameters():
    print(name, param.shape)

# %%
print([name for name, _ in model.model.named_children()])

# %%
from transformers.utils.import_utils import is_kernels_available
print(is_kernels_available())

# %%
import inspect
from transformers.utils.import_utils import is_kernels_available
print(inspect.getsource(is_kernels_available))
# %%
from transformers.utils.import_utils import KERNELS_MIN_VERSION
print(KERNELS_MIN_VERSION)
# %%
import kernels
print(kernels.__version__)
