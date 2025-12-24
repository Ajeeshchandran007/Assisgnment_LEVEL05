import os
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast 
from config import VOCAB_SIZE, TOKENIZER_DIR, MODEL_CONFIG, NUM_SAMPLES

def load_training_data(subset_size=None):
    """Loads and filters high-quality Wikipedia articles for tokenizer training."""
    if subset_size is None:
        subset_size = NUM_SAMPLES
    
    print(f"Loading Wikipedia dataset ({subset_size * 2} samples for filtering)...")
    # Load 2x samples so we can filter for quality
    dataset = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en", 
        split=f"train[:{subset_size * 2}]"
    )
    
    # Filter for quality articles
    def filter_quality(example):
        text = example["text"]
        word_count = len(text.split())
        # Keep only articles with substantial, clean content
        return (
            len(text) > 500 and          # Minimum character length
            len(text) < 10000 and        # Not too long (avoid lists/tables)
            word_count > 100 and         # Minimum word count
            text.count('\n') < len(text) / 50 and  # Not too fragmented
            not text.startswith('[[') and # Avoid metadata-heavy articles
            text.count('==') < 20        # Avoid overly structured articles
        )
    
    print("Filtering for quality content...")
    filtered = dataset.filter(filter_quality)
    
    # Take only what we need
    actual_size = min(subset_size, len(filtered))
    filtered = filtered.select(range(actual_size))
    print(f"Filtered to {len(filtered)} quality samples")
    
    return (text for text in filtered["text"])

def train_and_save_tokenizer(corpus_iterator):
    """Trains a BPE tokenizer or loads existing one."""
    if os.path.exists(TOKENIZER_DIR):
        print(f"Tokenizer directory found at {TOKENIZER_DIR}. Loading existing tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
        
        # Verify tokenizer with proper decoding
        print("\n" + "="*60)
        print("Tokenizer Verification:")
        print("="*60)
        test_text = "The quick brown fox jumps over the lazy dog."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        print(f"Original: {test_text}")
        print(f"Decoded:  {decoded}")
        print(f"Match: {'✓ Perfect!' if test_text == decoded else '✗ Mismatch'}")
        print(f"Vocab size: {len(tokenizer)}")
        print("="*60 + "\n")
        
        return tokenizer
        
    print("Training new BPE tokenizer with ByteLevel encoding/decoding...")
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())
    
    # Set pre-tokenizer to ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Set decoder to ByteLevel to convert Ġ back to spaces
    tokenizer.decoder = decoders.ByteLevel()
    
    # Set post-processor to ByteLevel
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Train the tokenizer
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    
    print("Training tokenizer on corpus...")
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    
    # Save the tokenizer
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Wrap in HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        model_max_length=MODEL_CONFIG.n_positions
    )
    
    hf_tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"HuggingFace tokenizer saved to {TOKENIZER_DIR}")
    
    # Verify the tokenizer works correctly
    print("\n" + "="*60)
    print("Tokenizer Verification:")
    print("="*60)
    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = hf_tokenizer.encode(test_text)
    decoded = hf_tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {'✓ Perfect!' if test_text == decoded else '✗ Mismatch'}")
    print(f"Vocab size: {len(hf_tokenizer)}")
    
    # Additional test with special characters
    test_text2 = "Hello, world! How are you? I'm fine, thanks."
    encoded2 = hf_tokenizer.encode(test_text2)
    decoded2 = hf_tokenizer.decode(encoded2, skip_special_tokens=True)
    print(f"\nTest 2 Original: {test_text2}")
    print(f"Test 2 Decoded:  {decoded2}")
    print(f"Test 2 Match: {'✓ Perfect!' if test_text2 == decoded2 else '✗ Mismatch'}")
    print("="*60 + "\n")
    
    return hf_tokenizer

def tokenize_and_group_data(hf_tokenizer, block_size=512, num_samples=None):
    """Tokenizes dataset and creates overlapping chunks for better learning."""
    if num_samples is None:
        num_samples = NUM_SAMPLES
    
    print(f"Loading {num_samples * 2} examples for filtering and tokenization...")
    dataset = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en", 
        split=f"train[:{num_samples * 2}]"
    )
    
    # Apply same quality filter
    def filter_quality(example):
        text = example["text"]
        word_count = len(text.split())
        return (
            len(text) > 500 and
            len(text) < 10000 and
            word_count > 100 and
            text.count('\n') < len(text) / 50 and
            not text.startswith('[[') and
            text.count('==') < 20
        )
    
    print("Filtering dataset...")
    dataset = dataset.filter(filter_quality)
    
    # Take only what we need
    actual_size = min(num_samples, len(dataset))
    dataset = dataset.select(range(actual_size))
    print(f"Using {len(dataset)} quality samples for training")
    
    def tokenize_function(examples):
        return hf_tokenizer(examples["text"], return_attention_mask=False)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=100,
        remove_columns=["id", "url", "title", "text"]
    )
    
    print("Creating overlapping chunks for more effective training...")
    all_input_ids = []
    
    for example in tokenized_datasets:
        all_input_ids.extend(example['input_ids'])
    
    # Create overlapping chunks (50% overlap) to maximize data usage
    chunks = []
    stride = block_size // 2  # 50% overlap between chunks
    
    for i in range(0, len(all_input_ids) - block_size, stride):
        chunk = {
            'input_ids': all_input_ids[i:i + block_size],
            'labels': all_input_ids[i:i + block_size]
        }
        chunks.append(chunk)
    
    print(f"Created {len(chunks)} overlapping chunks of size {block_size}")
    
    lm_dataset = Dataset.from_dict({
        'input_ids': [chunk['input_ids'] for chunk in chunks],
        'labels': [chunk['labels'] for chunk in chunks]
    })
    
    # 90/10 train/eval split
    train_test_split = lm_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train set: {len(train_test_split['train'])} samples")
    print(f"Eval set: {len(train_test_split['test'])} samples")
    
    return train_test_split["train"], train_test_split["test"]