"""
QLoRA Fine-tuning for Llama 3.2 Sentiment Classification
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# DATASET: EXPANDED AND BALANCED SENTIMENT DATA
# ============================================================================
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

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Use 1B for T4 GPU efficiency
OUTPUT_DIR = "./llama-3.2-qlora-sentiment"
MAX_LENGTH = 256  # Increased for better context
BATCH_SIZE = 2    
EPOCHS = 3        

# ============================================================================
# QLORA FINE-TUNER CLASS
# ============================================================================
class QLoRAFineTuner:
    """
    Proper QLoRA implementation
    """
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Verify GPU availability
        self.check_gpu()
        
    def check_gpu(self):
        """Check GPU availability and specs"""
        print("\n" + "="*70)
        print("🔍 SYSTEM CHECK")
        print("="*70)
        
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ No GPU detected! QLoRA requires CUDA.\n"
                "In Google Colab: Runtime → Change runtime type → GPU (T4)"
            )
        
        self.device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU Available: {gpu_name}")
        print(f"✅ Total GPU Memory: {gpu_memory:.2f} GB")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print("✅ GPU cache cleared")
        print("="*70)
        
    def load_model_and_tokenizer(self):
        """
        Load model with QLoRA configuration:
        - 4-bit NF4 quantization
        - Double quantization
        - BFloat16 compute dtype
        """
        print("\n📦 Loading model with QLoRA (4-bit quantization)...")
        
        # Load tokenizer with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer
            )
        except Exception as e:
            print(f"⚠️ Fast tokenizer failed, trying slow tokenizer: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
        
        # FIX 1: Properly set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = "right"
        
        print(f"   Tokenizer loaded: {type(self.tokenizer).__name__}")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"   EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
        # QLoRA 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                      # Enable 4-bit loading
            bnb_4bit_quant_type="nf4",             # NormalFloat4 quantization
            bnb_4bit_compute_dtype=torch.bfloat16, # Compute dtype
            bnb_4bit_use_double_quant=True,        # Double quantization
        )
        
        print("   Loading base model (this may take 2-3 minutes)...")
        
        # FIX 2: Load model with proper error handling
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # Explicit dtype
                low_cpu_mem_usage=True,      # Memory optimization
            )
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Trying alternative loading method...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # FIX 3: Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True  # Enable gradient checkpointing
        )
        
        # FIX 4: Disable cache and set configs
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'pretraining_tp'):
            self.model.config.pretraining_tp = 1
        
        # Print memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"✅ Model loaded successfully!")
        print(f"   GPU Memory Allocated: {memory_allocated:.2f} GB")
        print(f"   GPU Memory Reserved: {memory_reserved:.2f} GB")
        
    def setup_lora(self):
        """
        Setup LoRA adapters for QLoRA fine-tuning
        """
        print("\n🎯 Setting up LoRA configuration...")
        
        # FIX 5: Use safer target modules
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        
        lora_config = LoraConfig(
            r=8,                            # REDUCED rank for stability (was 16)
            lora_alpha=16,                  # Alpha = 2*r
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,           # Training mode
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # FIX 6: Enable gradient checkpointing for LoRA model
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        
        # Print trainable parameters
        print("\n📊 Trainable Parameters:")
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, data):
        """
        Prepare and tokenize dataset with instruction formatting
        """
        print("\n📊 Preparing dataset...")
        
        # FIX 7: Simpler prompt format for better results
        formatted_data = []
        for text, label in zip(data['text'], data['label']):
            # Simplified format - more reliable
            prompt = f"Classify the sentiment:\nText: {text}\nSentiment: {label}"
            formatted_data.append({'text': prompt})
        
        # Stratified train/validation split
        train_data, val_data = train_test_split(
            formatted_data,
            test_size=0.2,  # REDUCED validation size (was 0.25)
            random_state=42,
            stratify=data['label']
        )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # FIX 8: Improved tokenization function
        def tokenize_function(examples):
            # Tokenize with proper settings
            tokenized = self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None,  # Don't return tensors yet
            )
            
            # Set labels (for causal LM, labels = input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize datasets
        print("   Tokenizing training data...")
        self.train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        
        print("   Tokenizing validation data...")
        self.val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation"
        )
        
        # FIX 9: Set format for PyTorch
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")
        
        print(f"✅ Training samples: {len(self.train_dataset)}")
        print(f"✅ Validation samples: {len(self.val_dataset)}")
        
    def train(self):
        """
        Train model with QLoRA on T4 GPU
        """
        print("\n🚀 Starting QLoRA training...")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Gradient accumulation: 4")
        print(f"   Effective batch size: {BATCH_SIZE * 4}")
        print(f"   Epochs: {EPOCHS}")
        
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,              # Added weight decay
            fp16=False,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch",
            warmup_ratio=0.1,               # Warmup ratio instead of steps
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            dataloader_pin_memory=False,    # Disable pin memory for stability
            remove_unused_columns=False,    # Keep all columns
        )
        
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Optimize for GPU
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("\n⏳ Training in progress...")
        print("   (This may take 5-10 minutes on T4 GPU)")
        
        try:
            trainer.train()
            print("\n✅ Training completed successfully!")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            raise
        
        # Save model
        print("\n💾 Saving model...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✅ Model saved to: {self.output_dir}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
    def inference(self, text):
        """
        Run inference on new text
        """
        
        prompt = f"Classify the sentiment:\nText: {text}\nSentiment:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # Greedy decoding
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract sentiment from response
        response_lower = response.lower()
        
        # Look for sentiment in generated text
        if "sentiment:" in response_lower:
            sentiment_part = response_lower.split("sentiment:")[-1].strip()
        else:
            sentiment_part = response_lower
        
        # Match sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_part[:20]:  # Check first 20 chars
                return sentiment
        
        return "unknown"


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🎬 LLAMA 3.2 QLORA FINE-TUNING")
    print("   Sentiment Classification with 4-bit Quantization")
    print("   FIXED VERSION - Enhanced Stability")
    print("="*70)
    print(f"📊 Dataset: {len(data['text'])} samples")
    print(f"   ✅ Positive: {data['label'].count('positive')}")
    print(f"   ❌ Negative: {data['label'].count('negative')}")
    print(f"   ⚪ Neutral:  {data['label'].count('neutral')}")
    print("="*70)
    
    try:
        # Initialize fine-tuner
        fine_tuner = QLoRAFineTuner(MODEL_NAME, OUTPUT_DIR)
        
        # Load model with QLoRA
        fine_tuner.load_model_and_tokenizer()
        
        # Setup LoRA adapters
        fine_tuner.setup_lora()
        
        # Prepare dataset
        fine_tuner.prepare_dataset(data)
        
        # Train model
        fine_tuner.train()
        
        # ====================================================================
        # TESTING
        # ====================================================================
        print("\n" + "="*70)
        print("🧪 TESTING FINE-TUNED MODEL")
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
        
        print("\n🔍 Predictions:")
        print("-" * 70)
        
        for text, expected in test_cases:
            try:
                predicted = fine_tuner.inference(text)
                is_correct = "✅" if predicted == expected else "❌"
                if predicted == expected:
                    correct += 1
                
                print(f"{is_correct} Text: {text[:50]}...")
                print(f"   Expected: {expected.upper()} | Predicted: {predicted.upper()}\n")
            except Exception as e:
                print(f"❌ Error predicting: {e}\n")
        
        accuracy = (correct / total) * 100
        print("="*70)
        print(f"🎯 FINAL ACCURACY: {correct}/{total} = {accuracy:.1f}%")
        print("="*70)
        
        # ====================================================================
        # INTERACTIVE TESTING (Optional)
        # ====================================================================
        print("\n💬 Want to test interactively? (y/n): ", end="")
        if input().lower() == 'y':
            print("\nInteractive Testing Mode (type 'quit' to exit)\n")
            
            while True:
                user_input = input("Enter text: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                if user_input:
                    try:
                        sentiment = fine_tuner.inference(user_input)
                        print(f"🎭 Predicted Sentiment: {sentiment.upper()}\n")
                    except Exception as e:
                        print(f"❌ Error: {e}\n")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Ensure GPU is enabled: Runtime → Change runtime type → T4 GPU")
        print("   2. Install packages: !pip install -U transformers peft bitsandbytes")
        print("   3. Restart runtime if needed")
        print("   4. Check CUDA version compatibility")
        print("   5. Try reducing BATCH_SIZE to 1")