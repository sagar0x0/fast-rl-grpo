from torch.autograd import profiler
from torch.autograd.profiler import record_function
import torch.cuda.nvtx as nvtx

import os
# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Enable Hugging Face Hub transfer acceleration
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


import re
import random
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import List, Tuple
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

from datasets import load_dataset

# custom cross entropy kernel
from fused_ce import FusedCrossEntropy


# Enable TensorCores (Huge speedup for MatMul on Ampere+ GPUs)
torch.set_float32_matmul_precision('high')

# config
Q_batch_size = 1
num_gen_Q = 4
model_path = "meta-llama/Llama-3.2-1B-Instruct"
max_prompt_length = 300
max_new_token_len = 256
compute_gen_logps = True
all_steps = 500
save_steps = 10
lr = 5e-7
beta = 0.04
clip_param = 0.2
num_iterations = 4 # GRPO inner loop iterations


from huggingface_hub import login
try:
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
except:
    print("token not found")


# load models and tokenizers
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading ref model..")
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype = torch.bfloat16,
    _attn_implementation="sdpa",
    device_map = "auto",
)

# freeze ref_model eval/inference only 
ref_model.eval()
ref_model.requires_grad_(False)


print("Loading trainable policy model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype = torch.bfloat16,
    _attn_implementation="sdpa",
    device_map = "auto",
)

# torch.compile()  the model
model.compile()

# train the policy model
model.train()

# optimize the policy model states
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)


# load dataset
# path: "openai/gsm8k",   name: "main"
dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [ {'Q': q, 'A': a.split('####')[1].strip()}
       for q, a in zip(dataset['question'], dataset['answer'])]

print(f"loaded dataset of: {len(QAs)} example")


# utility function

# Optimized Log Prob Calculation (Speeds up forward/backward logic)
def get_per_token_logps(logits, input_ids):
    # Use CrossEntropyLoss (fused C++ kernel) instead of log_softmax + gather (Python loop)
    # logits: [B, L, V] -> [B*L, V] for CrossEntropy
    logits_flat = logits.reshape(-1, logits.size(-1))    # .reshape() : first check contigous then .view() on contiguous tensor
    input_ids_flat = input_ids.reshape(-1)         
    # loss_flat = -nn.CrossEntropyLoss(reduction='none')(logits_flat, input_ids_flat)
    # custom cross entropy kernel
    loss_flat = -FusedCrossEntropy.apply(logits_flat, input_ids_flat)
    return loss_flat.view(input_ids.shape)  # back to [B, L]

def get_ref_per_token_logps(input_ids, attention_mask, prompt_len):
    """input_ids: tokenized input,  prompt_len """
    with torch.inference_mode():
        # ref_model takes tokenized input_array the logits are accesed from output instance "SequenceClassifierOutput"
        outputs = ref_model(input_ids.to(ref_model.device), attention_mask=attention_mask.to(ref_model.device))
        logits = outputs.logits[:, :-1, :]
        shifted_ids = input_ids[:, 1:].to(ref_model.device)
        logpbs = get_per_token_logps(logits.to(torch.float32), shifted_ids)
    # detach so that no grad update flow to ref_model(just safety) as we already use torch.infernce()
    # no .cpu() off load as we have to load it again in grpo step 
    return logpbs[:, prompt_len - 1: ].detach()


# ===============
#Reward Functions
# ================

def parse_number(text):
    """ extract last number like token (decimal, fraction) """
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, text)
    return nums[-1] if nums else None

def save_eval(expr):
    """ save eval by converting fractions to float values """
    try:
        if '/' in expr:
            a, b = expr.split('/')
            return float(a) / float(b)
        return float(expr)
    except:
        return None

def reward_correct(item, answer):
    pred_num = parse_number(answer) # predicted value by model
    gt_num = parse_number(item["A"]) # ground truth value
    if pred_num is None or gt_num is None:
        return -1.0
    pred_val = save_eval(str(pred_num))
    gt_val = save_eval(str(gt_num))
    if pred_val is None or gt_val is None:
        return -1.0
    
    return 1.0 if abs(pred_val - gt_val) < 1e-5 else -1.0
    
def reward_format(answer):
    """ reward for correct format """
    pattern = r'^<think>.*?</think><answer>.*?</answer>$'
    return 1.0 if re.match(pattern, answer, re.DOTALL) else -1.0  # re.DOTALL makes . match newlines



# ==============================
# Generation (Policy Sampling)
# ==============================

system_prompt = """You are a helpful, precise, and honest assistant. Solve the user's math problem step by step. Enclose reasoning in <think> tags and final answer in <answer> tags, like:
<think> First, compute 3 * 4 = 12. Then add 5 to get 17. </think><answer>17</answer>
"""

generation_config = GenerationConfig(
    max_new_tokens=max_new_token_len,
    temperature=0.8,
    do_sample=True,   
    top_p=0.95,
    num_return_sequences=num_gen_Q,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

def gen_answers(prompts):
    # format prompt with chat template
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        chat_prompts.append(formatted_input)

    # encode with tokenizer
    inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1] # input_ids shape : (batch_size, prompt_length)
    if prompt_len > max_prompt_length:
        return [], 0

    # generation
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,   
            generation_config=generation_config,
        )
    
    # decode and extarct compeletion(remove prompt)
    completions = outputs[:, prompt_len:] 
    answer = []
    for comp in completions:
        # decode 
        text = tokenizer.decode(comp, skip_special_tokens=True)
        #clean
        text = re.sub(r"<\|.*?\|>", "", text).strip()
        answer.append(text)

    return answer, prompt_len

def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answer, prompt_len = gen_answers(prompts)

    # compute rewards
    rewards = []   # num_gen_Q rewards for each prompt
    for i, inp in enumerate(inputs):
        for a in answer[i * num_gen_Q: (i+1) * num_gen_Q]:
            r = reward_correct(inp, a) + reward_format(a)
            rewards.append(r)

    rewards = torch.tensor(rewards, dtype=torch.float32)

    # tokenize the prompts & completions
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "system_prompt"}, {"role": "user", "content": q}],
            tokenize = False,
            add_generation_prompt=True
        )
        for q in prompts
    ]

    prompt_inputs = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    )["input_ids"]

    comp_inputs = tokenizer(
        answer,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_new_token_len,
    )["input_ids"]

    return prompt_inputs, comp_inputs, rewards, answer


#======================
# GRPO(training step)
#======================
def GRPO_step(batch):
    prompt_length = batch["plen"]
    # non_blocking=True will cause DMA(direct memory access) from CPU pinned memory
    inputs = batch["inputs"].to(model.device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
    advantages = batch["rewards"].to(model.device, non_blocking=True).unsqueeze(1)   # (B, 1)

    # forward pass
    # pass attention mask to model to prevent attending to padding tokens
    logits = model(inputs, attention_mask=attention_mask).logits[:, :-1, :]   # (B, L-1, V) : 0,1,2....l-2
    input_ids = inputs[:, 1:]    # (B, L-1)  : 1,2,3.....l-1

    per_token_logps = get_per_token_logps(logits, input_ids)  # (B, L-1)
    per_token_logps = per_token_logps[:, prompt_length - 1: ]   # (B, completion only)  : last_prompt_tk, comp_0,comp_1,comp_2.....comp_l-1

    ref_logps = batch["refs"].to(per_token_logps.device)       # (B, comp_len)

    # kl divergence
    per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1

    # completion mask (ignoring token)
    comp_tokens = inputs[:, prompt_length: ]
    comp_mask = (comp_tokens != tokenizer.pad_token_id).float()

    # policy ratio & loss
    if "gen_logps" in batch and compute_gen_logps:
        gen_logps = batch["gen_logps"].to(model.device)
        ratio = torch.exp(per_token_logps - gen_logps)
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # Without old policy: use current logps as baseline (degenerate PPO)
        per_token_loss = per_token_logps * advantages
    
    # final loss: -(reward - beta * kl)
    per_token_loss = -(per_token_loss - beta * per_token_kl)

    loss = (per_token_loss * comp_mask).sum(dim=1) / (comp_mask.sum(dim=1) + 1e-8)
    
    return loss.mean() 


print("Starting Training")

for step in tqdm(range(1, all_steps + 1), desc="Training"):
    # generate a batch
    batch_inputs = random.sample(QAs, Q_batch_size)
    prompt_ids, comp_ids, rewards, answer = gen_samples(batch_inputs)

    if prompt_ids is None or len(rewards) == 0:
        print(f"skipped step: due to empty rewards/ids")
        continue

    # check rewards low variance || skip those batches
    if (rewards.max() - rewards.min()).item() < 0.01:
        print(f"skipped step: due to low reward variance")
        continue

    # normalize th rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # build a full sequence
    rep = comp_ids.shape[0] // prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    prompt_rep = prompt_ids.repeat(1, rep).view(-1, prompt_len)
    full_ids = torch.cat([prompt_rep, comp_ids], dim=1)

    # attention mask
    attention_mask = (full_ids != tokenizer.pad_token_id).long()

    # reference logprobs
    ref_logps = get_ref_per_token_logps(full_ids, attention_mask, prompt_len)

    # Optional: get current gen logps (for ratio)
    gen_logps = None
    if compute_gen_logps:
        with torch.inference_mode():
            logits = model(full_ids.to(model.device), attention_mask=attention_mask.to(model.device)).logits[:, :-1, :]
            shifted_ids = full_ids[:, 1:].to(model.device)
            # no .cpu() as we have to again load the tensor in gpu in next few line 
            gen_logps = get_per_token_logps(logits, shifted_ids)[:, prompt_len - 1: ].detach()
    
    # build batch dict
    batch = {
        "plen": prompt_len,
        "inputs": full_ids.pin_memory(), # still in cpu in pinned memory
        "attention_mask": attention_mask.pin_memory(), # still in cpu in pinned memory
        "rewards": rewards.pin_memory(), # still in cpu in pinned memory
        "refs": ref_logps      # in gpu 
    }

    if gen_logps is not None:
        batch["gen_logps"] = gen_logps # in gpu
    
    loss = torch.tensor(0.0)
    # training step in grpo

    for _ in range(num_iterations):
        loss = GRPO_step(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    
    # if step % 10 == 0:
    #    tqdm.write(f"Step {step} | Loss: {loss.item():.4f} | Rewards: {rewards.tolist()}")

    

print("Training finished")