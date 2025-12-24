## 📋 Overview


perform training of an LLM built from scratch using transformer library which is of reasonably sized using a suitable dataset such as Wikipedia text and test the LLM

This project trains a custom language model on Wikipedia data with:
- Custom BPE tokenizer with ByteLevel encoding
- 6-layer transformer architecture (~20M parameters)
- Mixed precision (FP16) training for efficiency
- Automated testing and interactive generation


###  **Train the model:**
```python
python train_llm.py
```

2. **Test generation:**
The script automatically tests the model after training with sample prompts.

3. **Interactive mode:**
Uncomment the last line in `train_llm.py` to enable interactive text generation.

## 📁 Project Structure

```
├── config.py           # Model configuration and hyperparameters
├── data_prep.py        # Data loading, tokenizer training, preprocessing
├── train_llm.py        # Main training script and testing functions
└── README.md           # This file
```



