
## 📋 Overview

implement transfer learning using an LLM that is used by all like Mistral or Claude with a small dataset for a chosen task

This project fine-tunes Llama 3.2-1B for sentiment classification using:
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **8-bit Quantization** to reduce memory usage
- **Balanced Dataset** with 60 examples (20 positive, 20 negative, 20 neutral)
- **Instruction-Based Prompting** for better classification performance



## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers peft datasets scikit-learn accelerate bitsandbytes
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- PyTorch with CUDA support

### Basic Usage

1. **Run the script:**
```bash
python TransferLearning.py
```

The script will automatically:
- Load Llama 3.2-1B model
- Apply LoRA configuration
- Train on sentiment dataset
- Evaluate on test cases
- Start interactive testing mode

2. **Use the trained model:**
```python
from TransferLearning import ImprovedLlamaFineTuner

fine_tuner = ImprovedLlamaFineTuner(MODEL_NAME, OUTPUT_DIR)
fine_tuner.load_model_and_tokenizer()

# Classify sentiment
sentiment = fine_tuner.inference("This product is amazing!")
print(sentiment)  # Output: positive
```

## 📊 Dataset

### Composition
- **Total Samples:** 60 (balanced across 3 classes)
- **Positive:** 20 examples
- **Negative:** 20 examples
- **Neutral:** 20 examples

### Split
- **Training:** 75% (45 samples)
- **Validation:** 25% (15 samples)
- **Stratified:** Maintains class balance in both sets

### Example Data Points
```python
Positive: "Outstanding quality and fast delivery!"
Negative: "Terrible experience, would not recommend."
Neutral: "Average product, does the job."
```

## ⚙️ Configuration

### Model Settings
```python
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-3.2-sentiment-v2"
MAX_LENGTH = 128
```

### LoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank of the update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[
        "q_proj",            # Query projection
        "k_proj",            # Key projection
        "v_proj",            # Value projection
        "o_proj"             # Output projection
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Hyperparameters
```python
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    fp16=True,              # Mixed precision training
    warmup_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch"
)
```

## 🏗️ Architecture

### LoRA Benefits
- **Trainable Parameters:** ~0.8M (only 0.06% of total model)
- **Memory Efficient:** Uses 8-bit quantization
- **Fast Training:** Only updates small adapter matrices
- **Preserves Base Model:** Original weights remain frozen

### Prompt Format
```
### Instruction:
Classify the sentiment as: positive, negative, or neutral

### Text:
[User input text here]

### Sentiment:
[Model generates: positive/negative/neutral]
```

## 📈 Performance

### Expected Results
- **Accuracy:** 80-100% on balanced test set
- **Training Time:** 5-10 minutes (5 epochs on T4 GPU)
- **Memory Usage:** ~6-8GB VRAM with 8-bit quantization

### Test Examples
```
✓ "This is incredible! Best thing ever!" → positive
✓ "Very disappointed with the quality." → negative
✓ "It's okay, nothing special." → neutral
✓ "Outstanding! Highly recommend!" → positive
```

## 🎯 Usage Examples

### 1. Training from Scratch

```python
from TransferLearning import ImprovedLlamaFineTuner

# Initialize
fine_tuner = ImprovedLlamaFineTuner(
    model_name="meta-llama/Llama-3.2-1B",
    output_dir="./my-sentiment-model"
)

# Load and configure
fine_tuner.load_model_and_tokenizer()
fine_tuner.setup_lora()

# Prepare your data
data = {
    'text': ["Great product!", "Terrible quality.", "It's okay."],
    'label': ['positive', 'negative', 'neutral']
}
fine_tuner.prepare_dataset(data)

# Train
fine_tuner.train()
```

### 2. Inference Only

```python
from TransferLearning import ImprovedLlamaFineTuner

# Load trained model
fine_tuner = ImprovedLlamaFineTuner(
    model_name="meta-llama/Llama-3.2-1B",
    output_dir="./llama-3.2-sentiment-v2"
)
fine_tuner.load_model_and_tokenizer()

# Classify
text = "This exceeded all my expectations!"
sentiment = fine_tuner.inference(text)
print(f"Sentiment: {sentiment}")
```

### 3. Batch Prediction

```python
texts = [
    "Amazing product, highly recommend!",
    "Waste of money, very disappointed.",
    "Average quality, nothing special."
]

for text in texts:
    sentiment = fine_tuner.inference(text)
    print(f"{text} → {sentiment}")
```

### 4. Interactive Testing

The script includes an interactive mode:
```
Interactive Testing (type 'quit' to exit):

Enter text: This is the best thing I've ever bought!
Predicted Sentiment: positive

Enter text: Not worth the price, very poor quality
Predicted Sentiment: negative

Enter text: quit
```

## 🔧 Customization

### Expand the Dataset

Add more examples to the `data` dictionary:
```python
data = {
    'text': [
        "Your positive example",
        "Your negative example",
        "Your neutral example",
        # ... add more
    ],
    'label': [
        'positive',
        'negative',
        'neutral',
        # ... corresponding labels
    ]
}
```

### Adjust LoRA Parameters

For more capacity (if needed):
```python
lora_config = LoraConfig(
    r=32,              # Increase rank
    lora_alpha=64,     # Increase alpha proportionally
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.05  # Adjust dropout
)
```

### Modify Training Duration

```python
training_args = TrainingArguments(
    num_train_epochs=10,      # Train longer
    learning_rate=2e-4,       # Adjust learning rate
    warmup_steps=20,          # More warmup
)
```

### Change Classification Categories

Modify the prompt and labels:
```python
# In prepare_dataset:
prompt = f"### Instruction:\nClassify as: urgent, normal, or low_priority\n\n### Text:\n{text}\n\n### Priority:\n{label}"

# Update your data labels:
data = {
    'label': ['urgent', 'normal', 'low_priority', ...]
}
```

