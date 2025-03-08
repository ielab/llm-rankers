from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
import re

def reward_func_setwise(completions, ground_truth, **kwargs):
    pattern = r'<think>.*?</think>\s*<answer>(.*?)</answer>'
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
    scores = []
    for match, label in zip(matches, ground_truth):
        if match:
            score = 0
            answer_text = match.group(1)
            if label == answer_text.strip():
                score += 1
            scores.append(score)
        else:
            scores.append(0)
    return scores


dataset = load_from_disk("msmarco-passage-setwise-r1")
model_name = "Qwen/Qwen2.5-3B-Instruct"
batch_size = 8
num_generations = 8

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


training_args = GRPOConfig(output_dir=f"checkpoints/{model_name}/GRPO",
                           learning_rate=1e-5,
                           bf16=True,
                           max_completion_length=2048,
                           max_prompt_length=4096,
                           logging_steps=10,
                           # use_vllm=True,
                           vllm_dtype="bfloat16",
                           per_device_train_batch_size=batch_size,
                           report_to='wandb',
                           num_generations=num_generations,
                           log_completions=True,
                           deepspeed='ds_zero0_config.json',
                           gradient_checkpointing=True,
                           )

trainer = GRPOTrainer(
    model=lora_model,
    reward_funcs=reward_func_setwise,
    args=training_args,
    train_dataset=dataset,
)
trainer.train(resume_from_checkpoint=False)