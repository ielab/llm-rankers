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


dataset_path = "reasonir-setwise-r1-v0.2"
model_name = "Qwen/Qwen3-32B"
batch_size = 2
num_generations = 16
num_iterations = 1 # so called on-policy
gradient_accumulation_steps = 4
learning_rate = 1e-5
temperature = 1.2
output_dir = f"checkpoints/{model_name}/GRPO-v0.2"
eval_steps = 10
save_steps = 20
kl_coeff = 0.001

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
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
)
lora_model = get_peft_model(model, lora_config)

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=learning_rate,
    fp16=True,
    max_completion_length=4096,
    max_prompt_length=6000,
    logging_steps=5,
    beta=kl_coeff,
    # use_vllm=True,
    # vllm_dtype="bfloat16",
    # vllm_gpu_memory_utilization=0.2,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    # eval_strategy='steps',
    report_to='wandb',
    num_generations=num_generations,
    log_completions=True,
    deepspeed='ds_zero0_config.json',
    gradient_checkpointing=True,
    temperature=temperature,
    num_iterations=num_iterations,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,
    save_steps=save_steps,
    # eval_steps=eval_steps,
    log_level='info',
)

trainer = GRPOTrainer(
    model=lora_model,
    reward_funcs=reward_func_setwise,
    args=training_args,
    train_dataset=load_from_disk(f"{dataset_path}/train"),
    # eval_dataset=load_from_disk(f"{dataset_path}/test"),
)
trainer.train(resume_from_checkpoint=False)

