# Results

## Evaluation Results  

| **Model**      | **Exact Match (EM) %** | **Execution Accuracy (EA) %** |
|---------------|----------------------|----------------------|
| **Base Model (LLaMA-2 7B)** | **39.60%** | _Not Calculated_ |
| **Fine-Tuned (LoRA Config 1)** | **63.59%** | **66.67%** |
| **Fine-Tuned (LoRA Config 2)** | **70.48%** | **75.00%** |


---

##  Fine-Tuning Configurations  

###  **First Fine-Tuning Run (LoRA Config 1)**  
**Adapter Parameters:**  
```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj", "o_proj", "q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=100,  # Change this if T4 Times out
    learning_rate=2e-4,
    fp16=True,  # Use mixed precision training
    logging_steps=1,
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    optim="adamw_hf",
    save_strategy="epoch",
    report_to="none"
)
```
Results:

Exact Match Accuracy: 63.59%
Execution Accuracy: 66.67%

Second Fine-Tuning Run (LoRA Config 2 – More Aggressive Training)
```python
config = LoraConfig(
    r=16,  # Increase rank for better adaptation
    lora_alpha=64,  # Higher alpha to scale weight updates
    target_modules=["k_proj", "o_proj", "q_proj", "v_proj"],
    lora_dropout=0.1,  # Slightly higher dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
args = TrainingArguments(
    per_device_train_batch_size=8,  # Increase batch size (may require more GPU memory)
    gradient_accumulation_steps=2,  # Less accumulation, more frequent updates
    warmup_steps=10,  # Longer warmup to stabilize learning
    num_train_epochs=3,  # More epochs for stronger training
    learning_rate=3e-4,  # Higher LR for faster adaptation
    fp16=True,  # Use mixed precision training
    logging_steps=10,
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    optim="adamw_torch",  # Try PyTorch's AdamW optimizer
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none",
    save_total_limit=2
)
```
Results:

Exact Match Accuracy: 70.48%
Execution Accuracy: 75.00%
Fine-tuning significantly improves SQL generation accuracy:

EM improved from 39.60% (base) → 70.48% (fine-tuned)
EA improved from N/A → 75.00%
