import torch
from transformers import Wav2Vec2ForCTC
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


# === Load quantized model ===
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
state_dict = torch.load("model_dyn.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)

# === Example input and dynamic shape spec ===
example_input = (torch.randn(5, 16000),)  # 5-second audio

# === Export ===
exported_program = export(model, example_input)

# === Transform & Lower ===
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]
).to_executorch()

# === Save as .pte ===
with open("model.pte", "wb") as file:
    executorch_program.write_to_file(file)
