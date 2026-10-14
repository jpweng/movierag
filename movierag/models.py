"""Loading the three models a RAG system needs.

1. Embedding model (bi-encoder): text -> vector. Used to index and to search.
2. Reranker (cross-encoder): (query, candidate) -> relevance score. Slower, sharper.
3. LLM (generator): reads the retrieved candidates and writes the answer.

The LLM is swappable behind LlamaIndex's `LLM` interface:
    hf          -> Qwen through Hugging Face transformers (like your workshop notebooks)
    ollama      -> any Ollama model on localhost:11434
    openai_like -> any OpenAI-compatible server (e.g. vLLM on Modal, see deploy/)
    mock        -> no model at all (offline plumbing tests)
"""

from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- embeddings
def load_embed_model(cfg):
    if cfg.offline:
        from llama_index.core import MockEmbedding

        return MockEmbedding(embed_dim=384)
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=cfg.embed_model, device=_device())


# ---------------------------------------------------------------- reranker
def load_reranker(cfg, top_n: int = 5):
    if cfg.offline:
        return None
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(model=cfg.rerank_model, top_n=top_n, device=_device())


# ---------------------------------------------------------------- LLM
def load_llm(cfg, backend: str | None = None, max_new_tokens: int = 700):
    backend = backend or cfg.llm_backend
    if backend == "mock":
        from llama_index.core.llms import MockLLM

        return MockLLM(max_tokens=80)
    if backend == "hf":
        return _load_hf(cfg, max_new_tokens)
    if backend == "ollama":
        from llama_index.llms.ollama import Ollama

        # thinking=False: Qwen3.5 "thinks" by default, which is slow and leaks into JSON.
        return Ollama(
            model=cfg.ollama_model,
            request_timeout=600,
            context_window=8192,
            temperature=0.0,
            thinking=False,
        )
    if backend == "openai_like":
        from llama_index.llms.openai_like import OpenAILike

        if not cfg.openai_like_base_url:
            raise ValueError("Set OPENAI_LIKE_BASE_URL (e.g. your Modal URL + '/v1').")
        return OpenAILike(
            model=cfg.openai_like_model,
            api_base=cfg.openai_like_base_url,
            api_key=cfg.openai_like_api_key,
            is_chat_model=True,
            context_window=32768,
            max_tokens=max_new_tokens,
            temperature=0.0,
            timeout=600,
            # vLLM forwards this to the chat template (disables Qwen "thinking").
            additional_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
    raise ValueError(f"Unknown LLM backend: {backend}")


def _load_hf(cfg, max_new_tokens: int):
    """Same calls as workshop notebook 03, plus 4-bit quantization so a 4B model
    fits in 8 GB of VRAM, then wrapped so LlamaIndex can drive it."""
    import torch
    from llama_index.llms.huggingface import HuggingFaceLLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs: dict = {"dtype": torch.bfloat16}
    if cfg.hf_load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # 4-bit "NormalFloat": bins shaped like a bell curve
            bnb_4bit_compute_dtype=torch.bfloat16,  # de-quantize to bf16 for the matmuls
            bnb_4bit_use_double_quant=True,  # also quantize the scaling constants
        )
    if not torch.cuda.is_available():
        model_kwargs["dtype"] = torch.float32
        log.warning("No CUDA GPU found: the LLM will run on CPU (slow). Consider LLM_BACKEND=ollama.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_model, device_map="auto", **model_kwargs)
    model.eval()

    def chat_template(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    return HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        model_name=cfg.hf_model,
        tokenizer_name=cfg.hf_model,
        context_window=8192,
        max_new_tokens=max_new_tokens,
        is_chat_model=True,  # use the model's chat template for chat() calls
        completion_to_prompt=chat_template,  # ...and for complete() calls too
        generate_kwargs={"do_sample": False, "repetition_penalty": 1.05},  # greedy = reproducible
    )


def free_gpu() -> None:
    """Give VRAM back after `del llm` (handy with only 8 GB)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_report() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU visible to PyTorch"
    free, total = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)}: {(total - free) / 2**30:.1f} GB used / {total / 2**30:.1f} GB"


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
