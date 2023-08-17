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
    embedding_file: str,
    tokenizer_path: str,
    max_gen_len: int = 256,
) -> str:
    # Create the ONNX session
    options = onnxruntime.SessionOptions()
    providers = onnxruntime.get_available_providers()
    providers = [provider for provider in providers if provider != "TensorrtExecutionProvider"]

    llm_session = onnxruntime.InferenceSession(
        onnx_file,
        sess_options=options,
        providers=providers,
    )

    update_kv_cache_session = onnxruntime.InferenceSession(
        "E:\\Llama-2-Onnx\\update_kv_cache.onnx",
        sess_options=options,
        providers=providers,
    )

    update_embeddings_session = onnxruntime.InferenceSession(
        "E:\\Llama-2-Onnx\\update_embeddings.onnx",
        sess_options=options,
        providers=providers,
    )

    argmax_sampling_session = onnxruntime.InferenceSession(
        "E:\\Llama-2-Onnx\\argmax_sampling.onnx",
        sess_options=options,
        providers=providers,
    )

    # get the data type used by the model
    data_type_str = llm_session.get_inputs()[0].type
    if data_type_str == "tensor(float16)":
        data_type = np.float16
    elif data_type_str == "tensor(float32)" or data_type_str == "tensor(float)":
        data_type = np.float32
    else:
        raise Exception(f"Unknown data type {data_type_str}")

    # Get the relevant shapes so we can create the inputs
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name == "x":
            x_shape = inputs_meta.shape
        elif inputs_meta.name == "attn_mask":
            attn_mask_shape = inputs_meta.shape
        elif inputs_meta.name == "k_cache":
            k_cache_shape = inputs_meta.shape

    hidden_size = x_shape[2]
    n_layers = k_cache_shape[1]
    n_heads = k_cache_shape[3]

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

    initial_cache_shape = (1, n_layers, 0, n_heads, head_dim)
    k_cache = onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache_shape, data_type, binding_device)
    v_cache = onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache_shape, data_type, binding_device)


    # Create the LLM model's I/O binding
    logits_shape = (1, tokenizer.n_words)
    logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(logits_shape, data_type, binding_device)
    llm_io_binding = llm_session.io_binding()
    llm_io_binding.bind_ortvalue_input("attn_mask", attn_mask)
    llm_io_binding.bind_ortvalue_output("logits", logits)

    # Create the KV cache's I/O binding
    update_kv_cache_io_binding = update_kv_cache_session.io_binding()
    update_kv_cache_io_binding.bind_ortvalue_output("k_cache_out", k_cache)
    update_kv_cache_io_binding.bind_ortvalue_output("v_cache_out", v_cache)

    # Create embeddings' I/O binding
    update_embeddings_io_binding = update_embeddings_session.io_binding()

    # Create the argmax sampling's I/O binding
    next_token = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1,), np.int64, binding_device)
    next_token_cpu = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1,), np.int64, "cpu")
    argmax_sampling_io_binding = argmax_sampling_session.io_binding()
    argmax_sampling_io_binding.bind_ortvalue_output("next_token", next_token)
    argmax_sampling_io_binding.bind_ortvalue_output("next_token_cpu", next_token_cpu)

    prev_seq_len = 0
    before_time = time.perf_counter()

    # Iteratively generate tokens.
    pos = np.array(0)
    output_tokens = []
    for idx in range(max_gen_len):
        seq_len = tokens.shape()[0]

        update_embeddings_io_binding.bind_ortvalue_input("tokens", tokens)

        # If the sequence length changed, we need to throw away the buffer and create a new one
        if seq_len != prev_seq_len:
            out_shape = (1, n_layers, seq_len, n_heads, head_dim)
            embeddings_shape = (1, seq_len, hidden_size)
            embeddings = onnxruntime.OrtValue.ortvalue_from_shape_and_type(embeddings_shape, data_type, binding_device)
            k_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(out_shape, data_type, binding_device)
            v_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(out_shape, data_type, binding_device)
            llm_io_binding.bind_ortvalue_input("x", embeddings)
            llm_io_binding.bind_ortvalue_output("k_out", k_out)
            llm_io_binding.bind_ortvalue_output("v_out", v_out)
            update_kv_cache_io_binding.bind_ortvalue_input("k_out", k_out)
            update_kv_cache_io_binding.bind_ortvalue_input("v_out", v_out)
            update_embeddings_io_binding.bind_ortvalue_output("embeddings", embeddings)
            prev_seq_len = seq_len

        # Update the embeddings
        update_embeddings_session.run_with_iobinding(update_embeddings_io_binding)

        # Run the LLM model
        llm_io_binding.bind_ortvalue_input("k_cache", k_cache)
        llm_io_binding.bind_ortvalue_input("v_cache", v_cache)
        llm_io_binding.bind_cpu_input("pos", pos.astype(np.int64))
        llm_session.run_with_iobinding(llm_io_binding)

        # Decide the next token using your preferred sampling strategy.
        argmax_sampling_io_binding.bind_ortvalue_input("logits", logits)
        argmax_sampling_session.run_with_iobinding(argmax_sampling_io_binding)
        output_tokens.append(next_token_cpu.numpy().item())

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if output_tokens[-1] == tokenizer.eos_id:
            break

        # Update the cache
        update_kv_cache_io_binding.bind_ortvalue_input("k_cache", k_cache)
        update_kv_cache_io_binding.bind_ortvalue_input("v_cache", v_cache)
        update_kv_cache_io_binding.bind_output("k_cache_out", binding_device)
        update_kv_cache_io_binding.bind_output("v_cache_out", binding_device)
        update_kv_cache_session.run_with_iobinding(update_kv_cache_io_binding)
        k_cache, v_cache = update_kv_cache_io_binding.get_outputs()[:2]

        # Update pos and x ready for the next round.
        pos = np.array(int(pos) + seq_len, dtype=np.int64)
        tokens = next_token

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
        "--embedding_file",
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
        args.embedding_file,
        args.tokenizer_path,
        args.max_gen_len,
    )

    print(response)
