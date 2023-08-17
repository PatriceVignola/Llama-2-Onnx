import argparse
import torch

class ArgmaxSampling(torch.nn.Module):
    def forward(self, logits):
        next_token = torch.argmax(logits, dim=-1)
        return next_token, next_token


def convert_model(vocab_size: int):
    logits = torch.randn((1, vocab_size), dtype=torch.float16)
    torch_model = ArgmaxSampling()
    torch_model.eval()

    torch.onnx.export(
        torch_model,
        logits,
        "argmax_sampling.onnx",
        do_constant_folding=True,
        input_names = ["logits"],
        output_names = ["next_token", "next_token_cpu"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_size",
        type=str,
        default=32000,
    )
    args = parser.parse_args()

    convert_model(args.vocab_size)
