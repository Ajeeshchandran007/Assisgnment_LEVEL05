# Fine-tuning Llama 3.2 for Multi-class Text Classification

A complete implementation for fine-tuning Meta's Llama 3.2 (1B) model for customer support ticket classification 

## 📋 Overview

This project demonstrates how to fine-tune Llama 3.2 for classifying customer support tickets into five categories:
- **Account** - Login, password, profile issues
- **Billing** - Payment, refunds, charges
- **Product** - Quality, defects, returns
- **Shipping** - Delivery, tracking, logistics
- **Technical** - Software bugs, errors, crashes

## ✨ Key Features

- **Optimized for Google Colab T4 GPU** 
- **4-bit Quantization** - Memory-efficient training using BitsAndBytes
- **LoRA Fine-tuning** - Parameter-efficient training (trains only 2-3% of parameters)
- **Expanded Dataset** - 120 training examples (24 per category)
- **Improved Prompting** - Clean, simple instruction format
- **Robust Evaluation** - Accuracy metrics, classification report, confusion matrix
- **Interactive Testing** - Easy-to-use prediction function


### Installation

The notebook automatically installs all required packages:
```python
transformers>=4.56.1
datasets
peft
accelerate
bitsandbytes
scikit-learn
trl
```


**Training Time:** ~10-15 minutes on T4 GPU

## 📁 Output Files

The notebook saves the fine-tuned model to:
```
./llama-classifier-finetuned/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── tokenizer.json
└── special_tokens_map.json
```

You can download these files from Colab's file browser.

## 🔧 Customization

### Adding Your Own Data

Modify the `data` dictionary in Cell 3:
```python
data = {
    'ticket': [
        "Your ticket text 1",
        "Your ticket text 2",
        # ... more examples
    ],
    'category': [
        'Category1',
        'Category2',
        # ... corresponding categories
    ]
}
```

### Changing Categories

Update the `CATEGORIES` list:
```python
CATEGORIES = ['YourCategory1', 'YourCategory2', ...]
```
