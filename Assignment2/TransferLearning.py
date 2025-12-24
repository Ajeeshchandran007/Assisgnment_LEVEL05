"""
Transfer Learning with Llama 3.2 for Text Classification - CPU Version
Optimized for CPU-only environments
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# EXPANDED and BALANCED dataset for sentiment analysis
data = {
    'text': [
        # Positive examples (20)
        "This product is amazing! I love it so much.",
        "Outstanding quality and fast delivery!",
        "Fantastic! Will buy again.",
        "Very satisfied with this purchase.",
        "Great value for money!",
        "Superb quality, highly recommend!",
        "Excellent product, works perfectly!",
        "Amazing! Best decision ever!",
        "This is incredible! Best thing ever!",
        "Exceeded my expectations, very happy!",
        "Absolutely love this! Worth every penny.",
        "Perfect! Exactly what I needed.",
        "Brilliant product, five stars!",
        "Best purchase I've made in years.",
        "Wonderful experience, highly satisfied.",
        "Outstanding service and quality.",
        "I'm thrilled with this product!",
        "Exceptional quality, can't fault it.",
        "This exceeded all my expectations!",
        "Absolutely perfect in every way.",
        
        # Negative examples (20)
        "Terrible experience, would not recommend.",
        "Absolutely horrible, waste of money.",
        "Worst purchase I've ever made.",
        "Disappointing quality for the price.",
        "Poor customer service and product.",
        "Below average, not worth it.",
        "Unsatisfactory, expected better.",
        "Very disappointed with the quality.",
        "Terrible quality, broke immediately.",
        "Waste of time and money.",
        "Awful product, requesting refund.",
        "Completely useless, very disappointed.",
        "Poor quality materials, falling apart.",
        "Not as described, very unhappy.",
        "Horrible experience, never again.",
        "Defective product, terrible service.",
        "Extremely disappointed, total waste.",
        "Very poor quality for the price.",
        "Broken on arrival, terrible packaging.",
        "Regret this purchase completely.",
        
        # Neutral examples (20)
        "Pretty good, met my expectations.",
        "Not bad, but could be better.",
        "Mediocre at best, nothing special.",
        "Okay, nothing to write home about.",
        "It's okay, nothing special.",
        "Average product, does the job.",
        "Decent but nothing extraordinary.",
        "Standard quality, as expected.",
        "It's fine, works as advertised.",
        "Acceptable product, no complaints.",
        "Fair quality for the price.",
        "Neither good nor bad, just average.",
        "Satisfactory but unremarkable.",
        "Standard product, nothing unique.",
        "Adequate for basic needs.",
        "Does what it says, nothing more.",
        "Reasonable quality, fair price.",
        "Average experience overall.",
        "It's okay, meets basic requirements.",
        "Neutral feelings about this purchase.",
    ],
    'label': (
        ['positive'] * 20 + 
        ['negative'] * 20 + 
        ['neutral'] * 20
    )
}

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-3.2-sentiment-cpu"
MAX_LENGTH = 128

class CPULlamaFineTuner:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cpu"
        print(f"Using device: {self.device}")
        print("⚠️  Running on CPU - training will be slower but works without GPU")
        
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer for CPU"""
        print("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model WITHOUT quantization for CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True  # Optimize CPU memory usage
        )
        
        # Move model to CPU explicitly
        self.model = self.model.to(self.device)
        print("Model and tokenizer loaded successfully!")
        
    def setup_lora(self):
        """Setup LoRA configuration"""
        print("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            r=8,  # Reduced rank for CPU efficiency
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Fewer modules for CPU
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, data):
        """Prepare dataset with improved prompt formatting"""
        print("Preparing dataset...")
        
        formatted_data = []
        for text, label in zip(data['text'], data['label']):
            prompt = f"### Instruction:\nClassify the sentiment as: positive, negative, or neutral\n\n### Text:\n{text}\n\n### Sentiment:\n{label}"
            formatted_data.append({'text': prompt})
        
        train_data, val_data = train_test_split(
            formatted_data,
            test_size=0.25,
            random_state=42,
            stratify=data['label']
        )
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
        
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
    def train(self):
        """Train with CPU-optimized hyperparameters"""
        print("Starting training...")
        print("⏳ This will take longer on CPU - please be patient")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,  # Reduced epochs for CPU
            per_device_train_batch_size=1,  # Small batch for CPU
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,  # Compensate with accumulation
            learning_rate=3e-4,
            fp16=False,  # Disable fp16 for CPU
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_steps=5,
            warmup_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=0,  # Important for CPU
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model(self.output_dir)
        print(f"Model saved to {self.output_dir}")
        
    def inference(self, text):
        """Run inference"""
        prompt = f"### Instruction:\nClassify the sentiment as: positive, negative, or neutral\n\n### Text:\n{text}\n\n### Sentiment:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment_part = response.split("### Sentiment:\n")[-1].strip().lower()
        sentiment = sentiment_part.split()[0] if sentiment_part else "unknown"
        sentiment = sentiment.rstrip(',.!?;:')
        
        return sentiment

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("LLAMA 3.2 SENTIMENT CLASSIFICATION - CPU VERSION")
    print("="*70)
    print(f"Dataset size: {len(data['text'])} samples")
    print(f"Balanced: {data['label'].count('positive')} positive, "
          f"{data['label'].count('negative')} negative, "
          f"{data['label'].count('neutral')} neutral")
    print("="*70 + "\n")
    
    fine_tuner = CPULlamaFineTuner(MODEL_NAME, OUTPUT_DIR)
    fine_tuner.load_model_and_tokenizer()
    fine_tuner.setup_lora()
    fine_tuner.prepare_dataset(data)
    fine_tuner.train()
    
    # Testing
    print("\n" + "="*70)
    print("TESTING MODEL")
    print("="*70)
    
    test_cases = [
        ("This is incredible! Best thing ever!", "positive"),
        ("Very disappointed with the quality.", "negative"),
        ("It's okay, nothing special.", "neutral"),
        ("Absolutely love this product!", "positive"),
        ("Terrible, complete waste of money.", "negative"),
        ("Average quality, does the job.", "neutral"),
        ("Outstanding! Highly recommend!", "positive"),
        ("Poor quality, not satisfied.", "negative"),
        ("Decent product, nothing special.", "neutral"),
        ("Perfect! Exactly what I wanted!", "positive"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print("\nPredictions:")
    print("-" * 70)
    
    for text, expected in test_cases:
        predicted = fine_tuner.inference(text)
        is_correct = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        
        print(f"{is_correct} Text: {text}")
        print(f"  Expected: {expected} | Predicted: {predicted}")
        print()
    
    accuracy = (correct / total) * 100
    print("="*70)
    print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")
    print("="*70)
    
    # Interactive testing
    print("\nInteractive Testing (type 'quit' to exit):")
    while True:
        user_input = input("\nEnter text: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input:
            sentiment = fine_tuner.inference(user_input)
            print(f"Predicted Sentiment: {sentiment}")