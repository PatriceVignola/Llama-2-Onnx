import torch

class UpdateKeyValueCache(torch.nn.Module):
    def forward(self, k_cache, v_cache, k_out, v_out):
        k_cache_out = torch.concat([k_cache, k_out], dim=2)
        v_cache_out = torch.concat([v_cache, v_out], dim=2)
        return k_cache_out, v_cache_out

pos = 0
seq_len = 1

k_cache = torch.randn(1, 32, pos, 32, 128, dtype=torch.float16)
v_cache = torch.randn(1, 32, pos, 32, 128, dtype=torch.float16)
k_out = torch.randn(1, 32, seq_len, 32, 128, dtype=torch.float16)
v_out = torch.randn(1, 32, seq_len, 32, 128, dtype=torch.float16)
torch_model = UpdateKeyValueCache()
torch_model.eval()

torch.onnx.export(
    torch_model,
    (k_cache, v_cache, k_out, v_out),
    "update_kv_cache.onnx",
    do_constant_folding=True,
    input_names = ["k_cache", "v_cache", "k_out", "v_out"],
    output_names = ["k_cache_out", "v_cache_out"],
    dynamic_axes={
        "k_cache": {2: "pos"},
        "v_cache": {2: "pos"},
        "k_out": {2: "seq_len"},
        "v_out": {2: "seq_len"},
        "k_cache_out": {2: "next_pos"},
        "v_cache_out": {2: "next_pos"},
    },
)