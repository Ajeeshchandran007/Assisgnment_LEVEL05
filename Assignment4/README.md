# QLoRA Fine-tuning for Llama 3.2 Sentiment Classification

A production-ready implementation of QLoRA (Quantized Low-Rank Adaptation) for fine-tuning Llama 3.2 on sentiment classification tasks using 4-bit quantization.

## 🎯 Features

- **Memory Efficient**: 4-bit NF4 quantization with double quantization
- **GPU Optimized**: Runs on Google Colab T4 GPU (~15GB VRAM)
- **Balanced Dataset**: 60 samples across positive, negative, and neutral sentiments
- **Enhanced Stability**: Multiple fixes for common training issues
- **Interactive Testing**: Built-in inference testing mode

## 📋 Requirements

```bash
pip install -U transformers peft bitsandbytes accelerate datasets scikit-learn
```



## 🚀 Quick Start

### Google Colab Setup

1. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator: GPU (T4)

2. **Install Dependencies**:
   ```python
   !pip install -q transformers peft bitsandbytes accelerate datasets scikit-learn
   ```

3. **Run the Script**:
   ```python
   python qlora_sentiment_finetuning.py
   ```

## 📊 Dataset

The training data consists of 60 balanced samples:
- **20 Positive** reviews (e.g., "Amazing! Best decision ever!")
- **20 Negative** reviews (e.g., "Terrible quality, broke immediately.")
- **20 Neutral** reviews (e.g., "Average product, does the job.")

Data is automatically split:
- **80%** Training (48 samples)
- **20%** Validation (12 samples)

## 🏗️ Architecture

### QLoRA Configuration
- **Quantization**: 4-bit NF4 with double quantization
- **Compute Dtype**: BFloat16
- **LoRA Rank**: 8 (reduced for stability)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## 📈 Training Process

1. **GPU Check**: Verifies CUDA availability and GPU specs
2. **Model Loading**: Loads Llama 3.2 with 4-bit quantization
3. **LoRA Setup**: Applies low-rank adapters to attention layers
4. **Dataset Preparation**: Tokenizes and formats data
5. **Training**: Fine-tunes for 3 epochs (~5-10 minutes on T4)
6. **Evaluation**: Tests on 10 sample cases
7. **Interactive Mode**: Optional manual testing

## 🧪 Testing

The script includes automated testing on 10 diverse examples:

```python
test_cases = [
    ("This is incredible! Best thing ever!", "positive"),
    ("Very disappointed with the quality.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    # ... 7 more cases
]
```

Expected accuracy: **80-100%** on test cases after training.

## 💬 Interactive Mode

After training, you can test custom inputs:

```
Enter text: This product exceeded all my expectations!
🎭 Predicted Sentiment: POSITIVE

Enter text: Not worth the price, very disappointing.
🎭 Predicted Sentiment: NEGATIVE
```
## 📁 Output Structure

```
./llama-3.2-qlora-sentiment/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── training_args.bin
```

