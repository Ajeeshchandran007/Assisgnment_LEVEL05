
from transformers import GPT2Config
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Verify GPU
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

TOKENIZER_DIR = "custom_llm_tokenizer_dir"
VOCAB_SIZE = 50257
NUM_SAMPLES = 50000  # Can use full 50K on GPU

# GPU-optimized model configuration - larger model since GPU can handle it
MODEL_CONFIG = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=512,      # GPU can handle longer context
    n_embd=384,           # larger embedding
    n_layer=6,            # more layers
    n_head=6,             # more attention heads
    n_inner=384 * 4,      # 1536 - feed-forward dimension
    activation_function="gelu",
    resid_pdrop=0.1,      # Standard dropout for GPU training
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
    pad_token_id=0
)