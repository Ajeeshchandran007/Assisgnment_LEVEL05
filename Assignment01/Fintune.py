"""
Fine-tuning Llama 3.2 for Multi-class Text Classification - 
Task: Customer Support Ticket Classification
Categories: Technical, Billing, Account, Shipping, Product
"""

# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================

# Check GPU availability
!nvidia-smi

print("\n" + "="*70)
print("Installing required packages...")
print("="*70)

# Install required packages
!pip install -q transformers>=4.56.1
!pip install -q datasets
!pip install -q peft
!pip install -q accelerate
!pip install -q bitsandbytes
!pip install -q scikit-learn
!pip install -q trl

print("✓ Installation complete!")

# ============================================================================
# CELL 2: Import Libraries and Setup
# ============================================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Verify GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*70}")
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*70}\n")

# ============================================================================
# CELL 3: Dataset Preparation 
# ============================================================================

# Dataset with more examples per category
data = {
    'ticket': [
        # Account issues (24 examples)
        "My account password is not working and I can't log in",
        "How do I change my email address on my account?",
        "I can't access the premium features I paid for",
        "Why was my account suspended without notification?",
        "How do I delete my account permanently?",
        "Cannot log in after password reset",
        "How do I change my account username?",
        "Need help setting up my account security questions",
        "How do I reset my two-factor authentication?",
        "My username is already taken, can you help?",
        "I forgot my security question answer",
        "Can I merge two accounts into one?",
        "My account shows wrong registration date",
        "How do I enable dark mode on my account?",
        "Need to verify my email address",
        "Account locked after too many login attempts",
        "How do I download my account data?",
        "Can I change my account region?",
        "Need to update my privacy settings",
        "How do I add a profile photo?",
        "My account name is misspelled",
        "Can I pause my account temporarily?",
        "How do I enable email notifications?",
        "Need help linking social media accounts",
        
        # Billing issues (24 examples)
        "I was charged twice for the same order on my credit card",
        "Why is there a pending charge on my statement?",
        "I want to update my billing information",
        "I haven't received my refund yet, it's been 10 days",
        "Automatic renewal charged me but I cancelled subscription",
        "Why was I charged a restocking fee?",
        "Unauthorized charges on my credit card",
        "I can't find my invoice for tax purposes",
        "What payment methods do you accept?",
        "Double charge on my statement",
        "Need to change my payment card",
        "Why is there a $1 authorization charge?",
        "Can I get a receipt for my purchase?",
        "My discount code wasn't applied",
        "Why am I being charged sales tax?",
        "Can I pay in installments?",
        "My refund amount is incorrect",
        "Need a VAT invoice",
        "Subscription price increased without notice",
        "Can I get a prorated refund?",
        "Payment declined but I have funds",
        "Currency conversion fee too high",
        "Need to cancel my recurring payment",
        "Can I change my billing cycle?",
        
        # Product issues (24 examples)
        "The product stopped working after 3 days of use",
        "The item I received is damaged and has scratches",
        "The product quality is not as described on the website",
        "I need to return a defective product",
        "The product manual is missing from the package",
        "Product arrived with missing accessories",
        "Received wrong item in my order",
        "The device overheats after 10 minutes of use",
        "Product size doesn't match description",
        "Colors don't match the website photos",
        "Item arrived expired",
        "Product makes strange noises",
        "Materials feel cheap and low quality",
        "Assembly instructions are unclear",
        "Product broke during first use",
        "Missing parts from the box",
        "Product doesn't fit as expected",
        "Warranty card is missing",
        "Item has manufacturing defect",
        "Product smells chemical",
        "Doesn't work with advertised devices",
        "Product arrived dirty",
        "Packaging was tampered with",
        "Serial number doesn't match",
        
        # Shipping issues (24 examples)
        "When will my order arrive? It's been 2 weeks already",
        "My package tracking shows it's stuck in transit",
        "Can I change my shipping address for order #12345?",
        "Package was delivered to wrong address",
        "My shipment is delayed by a week",
        "Shipping costs seem very high for my location",
        "Where is my tracking number?",
        "Package says delivered but I didn't receive it",
        "Can I expedite my shipping?",
        "My order shipped to old address",
        "Tracking hasn't updated in 5 days",
        "Package returned to sender",
        "Can I pick up from local facility?",
        "Shipping label has wrong name",
        "Package is stuck at customs",
        "Need to redirect my package",
        "Estimated delivery date passed",
        "Can I combine shipments?",
        "Package marked as lost",
        "Shipping insurance claim",
        "Need signature required delivery",
        "Package left in rain and damaged",
        "Can I schedule delivery time?",
        "Multiple packages, only received one",
        
        # Technical issues (24 examples)
        "The software keeps crashing when I try to export files",
        "Error code 500 appears when I try to upload documents",
        "Getting 'connection timeout' error constantly",
        "App freezes on startup every time",
        "Software license key is not activating",
        "The app won't sync with my account credentials",
        "Application crashes every time I click the save button",
        "Getting error 404 when accessing my dashboard",
        "Website won't load on mobile",
        "Can't download the app from store",
        "Features not working after update",
        "Getting SSL certificate error",
        "Videos won't play",
        "Search function returns no results",
        "Can't upload files larger than 5MB",
        "Page keeps refreshing automatically",
        "Images not displaying correctly",
        "Getting 'server error' message",
        "Can't connect to database",
        "Export function not working",
        "Mobile app won't install",
        "Buttons are unresponsive",
        "Screen goes blank randomly",
        "Audio playback issues",
    ],
    'category': (
        ['Account'] * 24 + 
        ['Billing'] * 24 + 
        ['Product'] * 24 + 
        ['Shipping'] * 24 + 
        ['Technical'] * 24
    )
}

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-classifier-finetuned"
MAX_LENGTH = 128  # Reduced for faster training
CATEGORIES = ['Account', 'Billing', 'Product', 'Shipping', 'Technical']

print("Dataset prepared!")
print(f"Total samples: {len(data['ticket'])}")
print(f"Categories: {', '.join(CATEGORIES)}")

# ============================================================================
# CELL 4: Helper Functions
# ============================================================================

def format_instruction(ticket, category=None):
    """Format data with simpler, more direct prompt"""
    if category:
        # Training format - clean and simple
        return f"""Classify this support ticket:
{ticket}
Category: {category}"""
    else:
        # Inference format
        return f"""Classify this support ticket:
{ticket}
Category:"""

def prepare_dataset():
    """Prepare and format the classification dataset"""
    df = pd.DataFrame(data)
    
    # Show class distribution
    print("\nClass Distribution:")
    print(df['category'].value_counts().sort_index())
    
    # Visualize distribution
    plt.figure(figsize=(10, 5))
    df['category'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Training Data Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Format as instruction-following
    df['formatted_text'] = df.apply(
        lambda row: format_instruction(row['ticket'], row['category']), 
        axis=1
    )
    
    # Create Hugging Face dataset
    dataset = Dataset.from_pandas(df[['formatted_text', 'category']])
    
    # Split into train and validation (80-20 split)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"\nTraining samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    return dataset

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    outputs = tokenizer(
        examples['formatted_text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors=None
    )
    outputs['labels'] = outputs['input_ids'].copy()
    return outputs

def plot_confusion_matrix(y_true, y_pred, categories):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

print("Helper functions loaded!")

# ============================================================================
# CELL 5: Load Model and Tokenizer
# ============================================================================

def setup_model_and_tokenizer():
    """Load and configure model with LoRA for T4 GPU"""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model with 4-bit quantization for T4 GPU...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=32,  # Increased rank for better capacity
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable, total = model.get_nb_trainable_parameters()
    print(f"\n✓ Model loaded successfully!")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    return model, tokenizer

model, tokenizer = setup_model_and_tokenizer()

# ============================================================================
# CELL 6: Prepare Training Data
# ============================================================================

print("Preparing dataset...")
dataset = prepare_dataset()

print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    remove_columns=['formatted_text']
)

print("✓ Data preparation complete!")

# ============================================================================
# CELL 7: Training Configuration
# ============================================================================

# Optimized training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,  # More epochs for better learning
    per_device_train_batch_size=8,  # Larger batch
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,  # 10% warmup
    learning_rate=5e-4,  # Higher learning rate
    fp16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    push_to_hub=False,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator
)

# Train the model
print("\n" + "="*70)
print("STARTING TRAINING ON T4 GPU")
print("="*70 + "\n")

trainer.train()

print("\n✓ Training complete!")

# Save model
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Model saved!")

# ============================================================================
# CELL 8: Evaluation
# ============================================================================

def extract_prediction(response):
    """Extract category from model response with multiple fallback methods"""
    
    # Method 1: Look for "Category:" pattern
    if "Category:" in response:
        after_category = response.split("Category:")[-1].strip()
        if after_category:
            words = after_category.split()
            if words:
                pred = words[0].strip('.,!?')
                # Validate it's one of our categories
                if pred in CATEGORIES:
                    return pred
    
    # Method 2: Look for any category word in the response
    response_upper = response.upper()
    for cat in CATEGORIES:
        if cat.upper() in response_upper:
            return cat
    
    # Method 3: Check last word
    words = response.strip().split()
    if words:
        last_word = words[-1].strip('.,!?')
        if last_word in CATEGORIES:
            return last_word
    
    return "Unknown"

def evaluate_model(model, tokenizer, test_dataset):
    """Evaluate the model on test set"""
    
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    model.eval()
    predictions = []
    ground_truth = []
    
    print("\nPredictions on test set:")
    print("-" * 70)
    
    for idx in range(len(test_dataset)):
        formatted = test_dataset[idx]['formatted_text']
        ticket = formatted.split("Classify this support ticket:")[1].split("Category:")[0].strip()
        true_category = test_dataset[idx]['category']
        
        prompt = format_instruction(ticket)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = extract_prediction(response)
        
        predictions.append(pred)
        ground_truth.append(true_category)
        
        status = "✓" if pred == true_category else "✗"
        print(f"{status} Ticket: {ticket[:50]}...")
        print(f"  True: {true_category:12} | Predicted: {pred}")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)
    
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")
    
    print("Detailed Classification Report:")
    print(classification_report(ground_truth, predictions, zero_division=0))
    
    # Plot confusion matrix
    plot_confusion_matrix(ground_truth, predictions, CATEGORIES)
    
    return predictions, ground_truth

predictions, ground_truth = evaluate_model(model, tokenizer, dataset['test'])

# ============================================================================
# CELL 9: Test on New Examples
# ============================================================================

def test_new_examples(model, tokenizer):
    """Test on completely new examples"""
    
    print("\n" + "="*70)
    print("TESTING ON NEW UNSEEN EXAMPLES")
    print("="*70)
    
    new_tickets = [
        "I forgot my password and the reset email isn't arriving",
        "The product broke within the warranty period",
        "Double charge appeared on my bank statement today",
        "Where is my package? Tracking hasn't updated in 5 days",
        "Application crashes every time I click the save button",
        "How do I update my profile picture?",
        "The item doesn't match the product description",
        "Why was I charged for expedited shipping?",
        "My order hasn't shipped yet and it's been a week",
        "Getting error 404 when accessing my dashboard"
    ]
    
    model.eval()
    results = []
    
    for ticket in new_tickets:
        prompt = format_instruction(ticket)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = extract_prediction(response)
        
        results.append({'ticket': ticket, 'prediction': prediction})
        print(f"\n📋 Ticket: {ticket}")
        print(f"🏷️  Predicted: {prediction}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("Summary of Predictions on New Examples:")
    print(results_df.to_string(index=False))
    
    return results_df

results_df = test_new_examples(model, tokenizer)

# ============================================================================
# CELL 10: Interactive Prediction
# ============================================================================

def predict_ticket(ticket_text):
    """Interactive prediction function"""
    model.eval()
    
    prompt = format_instruction(ticket_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = extract_prediction(response)
    
    print(f"\n{'='*70}")
    print("PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"\n📋 Your Ticket: {ticket_text}")
    print(f"🏷️  Predicted Category: {prediction}")
    print(f"\n{'='*70}")
    
    return prediction

# Example usage
print("\n" + "="*70)
print("TRY YOUR OWN TICKET!")
print("="*70)
print("\nExample: predict_ticket('Your support ticket text here')")
print("\nTry these examples:")
print("  predict_ticket('My credit card was charged multiple times')")
print("  predict_ticket('Cannot access my account dashboard')")
print("  predict_ticket('Product arrived broken and damaged')")

# Test one example
predict_ticket("The software license expired but I renewed it already")

print("\n" + "="*70)
print("✓ ALL DONE! Model is ready to use!")
print("="*70)
print(f"\nModel saved at: {OUTPUT_DIR}")
print("You can download it from Colab's file browser (left sidebar)")