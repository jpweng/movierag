"""Serve a bigger Qwen on a cloud GPU with vLLM, behind an OpenAI-compatible API.

Your laptop has 8 GB of VRAM; this runs a 30B mixture-of-experts model
(3B parameters active per token) on a 48 GB L40S, and scales to zero when idle.

One-time setup (in the project folder):
    uv sync --extra modal
    uv run modal setup                                        # log in
    uv run modal secret create movierag-llm VLLM_API_KEY=<pick-a-long-random-string>

Deploy (prints a URL like https://<you>--movierag-llm-serve.modal.run):
    uv run modal deploy deploy/modal_vllm.py

Then in .env:
    LLM_BACKEND=openai_like
    OPENAI_LIKE_BASE_URL=https://<you>--movierag-llm-serve.modal.run/v1
    OPENAI_LIKE_MODEL=qwen
    OPENAI_LIKE_API_KEY=<the same random string>

The first request after idle waits for a cold start (container + model load,
typically a few minutes the first time, faster once the weights are cached in
the Volume). You pay per second while a container is up.

NOTE: written against Modal 1.x and vLLM's `vllm serve` CLI. Pin the versions
you end up testing with; check the model card if you swap the model.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"  # text-only, no "thinking" mode, ~31 GB in FP8
GPU = "L40S"  # 48 GB. For an unquantized 30B model you'd need an 80 GB A100/H100.
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster weight downloads
)

# Volumes persist between runs, so weights are downloaded once, not on every cold start.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("movierag-llm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("movierag-llm")],  # provides VLLM_API_KEY
    scaledown_window=10 * 60,  # stay warm 10 min after the last request
    timeout=30 * 60,
)
@modal.concurrent(max_inputs=16)  # vLLM batches concurrent requests on one GPU
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", "qwen",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", "16384",       # plenty for RAG prompts, saves KV-cache memory
        "--gpu-memory-utilization", "0.90",
    ]
    # --api-key is read from the VLLM_API_KEY environment variable by vLLM.
    subprocess.Popen(cmd)


@app.local_entrypoint()
def smoke_test():
    """uv run modal run deploy/modal_vllm.py  -> sends one chat request."""
    import json
    import os
    import urllib.request

    url = serve.get_web_url() + "/v1/chat/completions"
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": "Name one great submarine movie, one line."}],
        "max_tokens": 60,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}",
    })
    with urllib.request.urlopen(req, timeout=1200) as resp:
        print(json.loads(resp.read())["choices"][0]["message"]["content"])
