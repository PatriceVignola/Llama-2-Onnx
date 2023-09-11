# This program will run the ONNX version of the LlamaV2 model.
import onnxruntime
import numpy as np
from sentencepiece import SentencePieceProcessor
from typing import List
import os
import argparse
import time


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


def get_binding_device_from_provider(provider: str) -> str:
    if provider == "DmlExecutionProvider":
        return "dml"

    if provider == "CUDAExecutionProvider":
        return "cuda"

    return "cpu"


def run_onnx_llamav2(
    prompt: str,
    onnx_file: str,
    update_embeddings_onnx_file: str,
    sampling_onnx_file: str,
    tokenizer_path: str,
    max_gen_len: int = 256,
) -> str:
    # Create the ONNX session
    providers = onnxruntime.get_available_providers()
    providers = [provider for provider in providers if provider != "TensorrtExecutionProvider"]

    options = onnxruntime.SessionOptions()
    update_embeddings_session = onnxruntime.InferenceSession(
        update_embeddings_onnx_file,
        sess_options=options,
        providers=providers,
    )

    argmax_sampling_session = onnxruntime.InferenceSession(
        sampling_onnx_file,
        sess_options=options,
        providers=providers,
    )

    # get the data type used by the model
    data_type = np.float16

    llm_session = onnxruntime.InferenceSession(
        onnx_file,
        sess_options=options,
        providers=providers,
    )

    # Get the relevant shapes so we can create the inputs
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name == "x":
            x_shape = inputs_meta.shape
        elif inputs_meta.name == "attn_mask":
            attn_mask_shape = inputs_meta.shape
        elif inputs_meta.name == "k_cache":
            cache_shape = inputs_meta.shape

    hidden_size = x_shape[2]
    n_layers = cache_shape[1]
    n_heads = cache_shape[3]
    max_seq_len = attn_mask_shape[1]

    binding_device = get_binding_device_from_provider(providers[0])

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = onnxruntime.OrtValue.ortvalue_from_numpy(np.asarray(tokens, dtype=np.int64), binding_device)

    # Create the attention mask.
    attn_mask = -10000.0 * np.triu(np.ones(attn_mask_shape), k=1).astype(data_type)
    attn_mask = onnxruntime.OrtValue.ortvalue_from_numpy(attn_mask, binding_device)

    # Create the K and V caches.
    head_dim = int(hidden_size / n_heads)

    cache_shape = (1, n_layers, max_seq_len, n_heads, head_dim)
    initial_cache = np.zeros(cache_shape, dtype=data_type)
    k_cache = onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, binding_device)
    v_cache = onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, binding_device)

    # Create the argmax sampling's I/O binding
    next_token = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1,), np.int64, binding_device)
    argmax_sampling_io_binding = argmax_sampling_session.io_binding()
    argmax_sampling_io_binding.bind_ortvalue_output("next_token", next_token)

    x = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, tokens.shape()[0], hidden_size), data_type, binding_device)
    x_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1, hidden_size), data_type, binding_device)
    cos = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len, 1, 64), data_type, binding_device)
    sin = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len, 1, 64), data_type, binding_device)

    # Create the LLM model's I/O binding
    logits_shape = (1, tokenizer.n_words)
    logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(logits_shape, data_type, binding_device)
    llm_io_binding = llm_session.io_binding()
    llm_io_binding.bind_ortvalue_output("logits", logits)
    llm_io_binding.bind_ortvalue_output("attn_mask_out", attn_mask)
    llm_io_binding.bind_ortvalue_output("k_out", k_cache)
    llm_io_binding.bind_ortvalue_output("v_out", v_cache)
    llm_io_binding.bind_ortvalue_output("cos_out", cos)
    llm_io_binding.bind_ortvalue_output("sin_out", sin)
    llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))

    update_embeddings_io_binding = update_embeddings_session.io_binding()
    update_embeddings_io_binding.bind_ortvalue_input("tokens", tokens)
    update_embeddings_io_binding.bind_ortvalue_output("embeddings", x)

    before_time = time.perf_counter()
    output_str = ""

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        # Update the embeddings
        update_embeddings_session.run_with_iobinding(update_embeddings_io_binding)
        update_embeddings_io_binding.synchronize_outputs()

        # Run the LLM model
        llm_io_binding.bind_ortvalue_input("x", x)
        llm_io_binding.bind_ortvalue_input("attn_mask", attn_mask)
        llm_io_binding.bind_ortvalue_input("k_cache", k_cache)
        llm_io_binding.bind_ortvalue_input("v_cache", v_cache)
        llm_io_binding.bind_ortvalue_input("x_increment", x_increment)
        llm_io_binding.bind_ortvalue_input("cos", cos)
        llm_io_binding.bind_ortvalue_input("sin", sin)
        llm_session.run_with_iobinding(llm_io_binding)
        llm_io_binding.synchronize_outputs()

        # Decide the next token using your preferred sampling strategy.
        argmax_sampling_io_binding.bind_ortvalue_input("logits", logits)
        argmax_sampling_session.run_with_iobinding(argmax_sampling_io_binding)
        argmax_sampling_io_binding.synchronize_outputs()
        output_tokens.append(next_token.numpy().item())

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if output_tokens[-1] == tokenizer.eos_id:
            break

        # Update the embeddings for the next iteration
        update_embeddings_io_binding.bind_ortvalue_input("tokens", next_token)

        if idx == 0:
            llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
            update_embeddings_io_binding.bind_ortvalue_output("embeddings", x_increment)

    after_time = time.perf_counter()
    print(f"Execution took {after_time - before_time:0.4f} seconds")

    output_str = tokenizer.decode(output_tokens)

    return output_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--onnx_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--update_embeddings_onnx_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sampling_onnx_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
    )
    parser.add_argument("--max_gen_len", type=int, default=256)
    args = parser.parse_args()
    response = run_onnx_llamav2(
        args.prompt,
        args.onnx_file,
        args.update_embeddings_onnx_file,
        args.sampling_onnx_file,
        args.tokenizer_path,
        args.max_gen_len,
    )

    print(response)
