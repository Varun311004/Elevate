"""Language model wrapper using HIGH-SPEED Quantized GGUF on CPU."""
from __future__ import annotations
import threading
from typing import Optional
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from config import MODELS_DIR

# We are using a 4-bit compressed version of Qwen 2.5 1.5B
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

class LanguageModel:
    def __init__(self) -> None:
        print(f"Downloading/Loading Quantized GGUF model: {REPO_ID}...")
        
        # 1. Download model to persistent cache (only downloads once)
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir=str(MODELS_DIR)
        )
        
        # 2. Load with llama.cpp optimized for CPU
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,          # Context window size
            n_threads=2,         # Max out the 2 vCPUs of HF Free Tier
            verbose=False        # Keep logs clean
        )
        
        # ADD THIS: A dedicated lock for the generation process
        self._inference_lock = threading.Lock()
        print("Quantized local model loaded successfully.")

    def ensure_model_loaded(self) -> None:
        pass

    def is_model_loaded(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_time: Optional[float] = None,
    ) -> str:
        # Format prompt for Qwen ChatML
        formatted_prompt = (
            f"<|im_start|>system\nYou are an expert educational AI. "
            f"Generate exactly what is requested, following the text format perfectly.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        
        # ADD THIS: Enforce single-file processing to prevent SegFaults
        with self._inference_lock:
            response = self.llm(
                prompt=formatted_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|im_end|>"]
            )
        
        return response["choices"][0]["text"].strip()

_llm_model: LanguageModel | None = None
_llm_model_lock = threading.Lock()

def get_llm_model() -> LanguageModel:
    global _llm_model
    if _llm_model is None:
        with _llm_model_lock:
            if _llm_model is None:
                _llm_model = LanguageModel()
    return _llm_model