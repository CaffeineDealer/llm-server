
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
inputs = tokenizer("The capital of France is", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}

hook.remove()  # disable hook
outputs_baseline = model.generate(**inputs, max_new_tokens=5)
print("BASELINE:", tokenizer.decode(outputs_baseline[0], skip_special_tokens=True))

# re-register hook
hook = model.model.language_model.layers[32].self_attn.register_forward_hook(make_gain_hook(gain=200.5))
outputs_boosted = model.generate(**inputs, max_new_tokens=5)
print("BOOSTED:", tokenizer.decode(outputs_boosted[0], skip_special_tokens=True))


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
