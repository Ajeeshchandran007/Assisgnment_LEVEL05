import os
import torch
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from config import DEVICE, MODEL_CONFIG, TOKENIZER_DIR
from data_prep import train_and_save_tokenizer, load_training_data, tokenize_and_group_data
import math

# For Google Colab - save to Google Drive
OUTPUT_DIR = "/content/drive/MyDrive/custom_llm_output" 
# Uncomment below for local training
# OUTPUT_DIR = "./custom_llm_output" 

def train_llm():
    """Main training function optimized for GPU."""
    print("="*60)
    print("Starting LLM Training Pipeline")
    print("="*60)
    
    # Verify GPU is available
    if DEVICE == "cuda":
        print(f"✓ GPU Training Enabled: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ WARNING: Running on CPU. Training will be very slow!")
    
    # Step 1: Load and prepare data
    corpus_iterator = load_training_data()
    tokenizer = train_and_save_tokenizer(corpus_iterator)
    
    # Step 2: Tokenize and create training datasets
    train_dataset, eval_dataset = tokenize_and_group_data(
        tokenizer, 
        block_size=MODEL_CONFIG.n_positions
    )
    
    # Limit eval dataset size for faster evaluation
    if len(eval_dataset) > 500:
        eval_dataset = eval_dataset.select(range(500))
        print(f"Limited eval dataset to 500 samples for efficiency")
    
    print(f"\nDataset Summary:")
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Eval dataset size: {len(eval_dataset)}")
    
    # Step 3: Initialize model
    print("\nInitializing model from custom config...")
    model = GPT2LMHeadModel(config=MODEL_CONFIG).to(DEVICE)
    total_params = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Model Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB (fp32)")
    
    # Step 4: Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )

    # Step 5: Configure training arguments (optimized for T4 GPU)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # Training schedule
        num_train_epochs=3,              # 3 epochs for GPU
        per_device_train_batch_size=16,  # Larger batch size for GPU
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,   # Effective batch size = 64
        
        # Learning rate and optimization
        learning_rate=5e-4,              # Standard LR
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        optim="adamw_torch",
        
        # Evaluation and logging
        eval_strategy="steps",
        eval_steps=500,                  # Evaluate every 500 steps
        logging_steps=50,                # Log every 50 steps
        logging_first_step=True,
        
        # Saving
        save_steps=500,                  # Save every 500 steps
        save_total_limit=3,              # Keep 3 checkpoints
        
        # GPU Performance optimization
        fp16=True,                       # Enable mixed precision for T4 GPU
        dataloader_num_workers=2,        # Parallel data loading
        dataloader_pin_memory=True,      # Pin memory for faster GPU transfer
        remove_unused_columns=False,
        
        # Evaluation settings
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        # Memory optimization
        gradient_checkpointing=False,    # Set to True if running out of memory
    )

    # Step 6: Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Calculate training info
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Total epochs: {training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Mixed Precision (FP16): {training_args.fp16}")
    print(f"  Estimated time: ~30-60 minutes on T4 GPU")
    print("="*60)
    
    # Step 7: Train the model
    print("\nStarting pre-training...")
    
    trainer.train()

    # Step 8: Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"Model and tokenizer saved to {OUTPUT_DIR}/final_model")
    
    # Step 9: Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*60)
    print("*** Final Evaluation Results ***")
    print("="*60)
    print(f"  Loss: {eval_results['eval_loss']:.4f}")
    
    try:
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"  Perplexity: {perplexity:.2f}")
        
        # Provide interpretation
        if perplexity < 50:
            print("  Quality: Excellent! Model has learned well.")
        elif perplexity < 150:
            print("  Quality: Good. Model shows reasonable performance.")
        elif perplexity < 500:
            print("  Quality: Fair. Model may need more training or data.")
        else:
            print("  Quality: Poor. Consider adjusting hyperparameters.")
    except:
        print("  (Could not calculate perplexity)")
    
    print("="*60)
    print("\nTraining complete!")

def test_llm():
    """Test the trained model with various prompts."""
    print("\n" + "="*60)
    print("Testing Text Generation")
    print("="*60)
    
    # Load model and tokenizer
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        print(f"Loading model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error: Could not load model. Please run training first.")
        print(f"Details: {e}")
        return

    # Test prompts
    prompts = [
        "The quick brown fox jumps over the",
        "In the beginning,",
        "Artificial intelligence is",
        "The history of",
        "Scientists have discovered"
    ]
    
    print("Generating text for test prompts...\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: \"{prompt}\"")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        # Generate with optimized parameters
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
            )

        # Decode and display
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}\n")
        print("-" * 60 + "\n")
    
    print("="*60)
    print("Text generation testing complete!")
    print("="*60)

def interactive_generation():
    """Interactive mode for testing custom prompts."""
    print("\n" + "="*60)
    print("Interactive Text Generation Mode")
    print("="*60)
    print("Type your prompts and press Enter. Type 'quit' to exit.\n")
    
    # Load model
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    while True:
        prompt = input("\nYour prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode.")
            break
        
        if not prompt:
            continue
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=100,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nGenerated: {generated_text}")

if __name__ == "__main__":
    # Run training
    train_llm()
    
    # Run automated tests
    test_llm()
    
    # Optional: Uncomment to enable interactive mode
    # interactive_generation()