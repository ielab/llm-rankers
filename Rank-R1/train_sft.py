from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

dataset = load_from_disk("msmarco-passage-setwise-sft")
model_name = "Qwen/Qwen2.5-3B-Instruct"
batch_size = 8
gradient_accumulation_steps = 8

tokenizer = AutoTokenizer.from_pretrained(model_name)

instruction_template = "<|im_start|>system"
response_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                           response_template=response_template,
                                           tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
)

lora_config = LoraConfig(
    base_model_name_or_path=model_name,
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)
lora_model = get_peft_model(model, lora_config)


training_args = SFTConfig(
    output_dir=f"checkpoints/{model_name}/SFT",
    max_length=4096,
    bf16=True,
    learning_rate=1e-5,
    logging_steps=10,
    per_device_train_batch_size=batch_size,
    report_to='wandb',
    deepspeed='ds_zero0_config.json',
    gradient_checkpointing=False,  # True gives error https://github.com/huggingface/trl/issues/2819
    packing=False,
    dataset_num_proc=40,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,
    ddp_timeout=18000
)

trainer = SFTTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train()