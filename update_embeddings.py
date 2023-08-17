import argparse
import torch

class UpdateEmbeddings(torch.nn.Module):
    def __init__(self, embedding_file: str, vocab_size: int, hidden_size: int):
        super(UpdateEmbeddings, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.embedding_layer.load_state_dict(torch.load(embedding_file))

    def forward(self, tokens):
        embeddings = self.embedding_layer(tokens)
        embeddings = torch.unsqueeze(embeddings, 0)
        return embeddings


def convert_model(embedding_file: str, vocab_size: int, hidden_size: int):
    seq_len = 8
    tokens = torch.zeros(seq_len, dtype=torch.int64)
    torch_model = UpdateEmbeddings(embedding_file, vocab_size, hidden_size)
    torch_model.eval()

    torch.onnx.export(
        torch_model,
        tokens,
        "update_embeddings.onnx",
        do_constant_folding=True,
        input_names = ["tokens"],
        output_names = ["embeddings"],
        dynamic_axes={
            "tokens": {0: "seq_len"},
            "embeddings": {1: "seq_len"},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vocab_size",
        type=str,
        default=32000,
    )
    parser.add_argument(
        "--hidden_size",
        type=str,
        default=4096,
    )
    args = parser.parse_args()

    convert_model(args.embedding_file, args.vocab_size, args.hidden_size)
