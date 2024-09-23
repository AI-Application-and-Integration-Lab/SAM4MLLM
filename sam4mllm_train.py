# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# %%
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.9"

import orjson
import warnings
from itertools import chain
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
from tqdm.auto import tqdm
from PIL import Image
from wandb.integration.lightning.fabric import WandbLogger

import torch
from lightning.fabric import Fabric

from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

# %%
PRETRAINED = "/root/autodl-tmp/big_models/llama3-llava-next-8b"

NUM_EPOCH = 3
BATCH_SIZE = 1
GRAD_ACC_STEPS = 8
MAX_LEN = 1536
USE_WANDB = True

# %%
class LlavaDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer,
                image_processor, model_config,
                max_len=1536,
            img_dir='/home/ai2lab/work/datasets/',
            img_size=(672,672),
        ):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.max_len = max_len
        self.img_dir = img_dir
        self.img_size = img_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, tokenize=True):
        d = self.data[idx]
        image_path = d['image_path']
        image = Image.open(self.img_dir + image_path).resize(self.img_size)
        
        conv_text = self.tokenizer.apply_chat_template(d['conversation'], tokenize=False)
        
        if tokenize:
            input_ids = tokenizer_image_token(conv_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids[:self.max_len]
        else:
            input_ids = conv_text
        
        pixel_values = process_images([image], self.image_processor, self.model_config)[0]
        image_sizes = list(self.img_size)

        sample = {
            'input_ids': input_ids,
            'images': pixel_values,
            'image_sizes': image_sizes
        }
        
        return sample
    

def load_processor():
    from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor
    
    model_config = AutoConfig.from_pretrained(PRETRAINED)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(PRETRAINED)
    
    return tokenizer, image_processor, model_config

def load_dataset(tokenizer, image_processor, model_config, max_len=1536):
    from data import GroupRefDataset
    
    with open('./refcoco_convs/refcoco_convs_ep1.json', 'r') as f:
        refcoco_data = orjson.loads(f.read())
        
    all_data = refcoco_data
        
    train_dataset = LlavaDataset(
        all_data,
        tokenizer,
        image_processor,
        model_config,
        max_len=max_len,
        img_dir='/root/autodl-tmp/img_datasets/',
        img_size=(672, 672),
    )
    print(f'num train examples: {len(train_dataset)}')
    
    return train_dataset

def load_model():
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.model import LlavaLlamaForCausalLM
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model, _, max_length = load_pretrained_model(
            PRETRAINED,
            None,
            "llava_llama3",
            # device_map=device_map,
            # device_map="auto",
            device_map=None,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            # quantization_config=quantization_config,
        )

    model.eval()
    model.tie_weights()
    
    return model

def get_lora_model(model):
    from peft import LoraConfig, get_peft_model
    
    target_modules = []
    for i in range(32):
        target_modules.append(f'model.layers.{i}.self_attn.q_proj')
        target_modules.append(f'model.layers.{i}.self_attn.k_proj')
        target_modules.append(f'model.layers.{i}.self_attn.v_proj')
        target_modules.append(f'model.layers.{i}.self_attn.o_proj')
        target_modules.append(f'model.layers.{i}.mlp.gate_proj')
        target_modules.append(f'model.layers.{i}.mlp.up_proj')
        target_modules.append(f'model.layers.{i}.mlp.down_proj')
    target_modules += ['mm_projector.0', 'mm_projector.2']

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # modules_to_save=["mm_projector"],
    )
    model = get_peft_model(model, peft_config)
    
    return model
    
def main(fabric):
    import torch
    # torch.set_default_dtype(torch.bfloat16)
    from torch.utils.data import Dataset, DataLoader

    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers import LlavaNextForConditionalGeneration
    from transformers import BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM
    from bitsandbytes.optim import AdamW8bit
    from lightning.pytorch import utilities as pt_utils
    
    tokenizer, image_processor, model_config = load_processor()
    train_dataset = load_dataset(tokenizer, image_processor, model_config)
    
    # testing
    d = train_dataset[0]
    
    instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    
    model = load_model(PRETRAINED)
    model = get_lora_model(model)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    
    LR = 1e-4
    total_steps = len(train_loader) // (GRAD_ACC_STEPS * fabric.world_size) * NUM_EPOCH
    if fabric.global_rank == 0:
        print(f"total steps: {total_steps}")
    
    optimizer = AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        pct_start=0.01,
        total_steps=total_steps,
        anneal_strategy="linear",
    )
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(train_loader)
    
    global_step = 0
    
    model.train()
    for epoch in range(NUM_EPOCH):
        batch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=fabric.global_rank != 0)
        for iteration, batch in enumerate(pbar):
            
            is_accumulating = iteration % GRAD_ACC_STEPS != 0

            loss = model(**batch, use_cache=False).loss
            fabric.backward(loss)
            batch_loss += loss.item()
                
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                fabric.log_dict({
                    "trainer/loss": batch_loss/GRAD_ACC_STEPS,
                    "trainer/lr": optimizer.param_groups[0]["lr"],
                    "trainer/epoch": epoch,
                    "trainer/global_step": global_step,
                })
                
                if fabric.global_rank == 0:
                    pbar.set_postfix({
                        "loss": batch_loss/GRAD_ACC_STEPS,
                        "lr": optimizer.param_groups[0]["lr"],
                        "global_step": global_step,
                    })
                    tqdm.write(f"loss: {batch_loss/GRAD_ACC_STEPS:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}, global_step: {global_step}")
                
                batch_loss = 0
                global_step += 1
                
            if (global_step+1) % 200 == 0 and fabric.global_rank == 0:
                save_dir = f"./refcoco_z2"
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = f"{save_dir}/checkpoint-{global_step}"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                
                print(f"Succesfully saved model at {save_path}")
            
            torch.cuda.empty_cache()

# %%
import torch
from lightning.fabric.strategies import DeepSpeedStrategy

# %%
logger = WandbLogger(project="sam4mllm")

fabric = Fabric(
    accelerator="cuda",
    devices=10,
    precision="bf16-true",
    strategy=DeepSpeedStrategy(
        zero_optimization=True,
        stage=2,
    ),
    loggers=logger,
)

fabric.launch(main)
