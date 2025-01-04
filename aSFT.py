## make all imports 
import torch
from torch import Tensor
import time
import sys
import json
from pathlib import Path
sys.path.append('/n/netscratch/pehlevan_lab/Everyone/indranilhalder/../LLM-SFT/') # add path to current directory

from LLM_aligned_SFT.asft import (
    aSFT,
    aSFTTrainer,
)
from LLM_aligned_SFT.dataset import create_example_dataset 

from x_transformers import TransformerWrapper, Decoder
import random
from datasets import load_dataset
from transformers import  AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from huggingface_hub import login  
login(token="..") # make sure to add correct huggingface tokens 

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# function to clear GPU cache
def clear_gpu_cache():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

# add memory cleanup between operations
clear_gpu_cache()

print("Custom addition dataset")
# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Running on CPU")
    
    
config_path = "config.json"
# read the hyperparameters from the config file 
with open(config_path, 'r') as fconfig:
    CONFIG = json.load(fconfig)

max_seq_len=CONFIG['max_seq_len']
length_check=int(max_seq_len/2) 
dataset_length=CONFIG['dataset_length']
heads=CONFIG['heads']
depth=CONFIG['depth']
decoder_dim=CONFIG['decoder_dim']
lSFT=CONFIG['lSFT']
lA=CONFIG['lA']
epochs=CONFIG['epochs']
batchsize=CONFIG['batchsize']
temperature=CONFIG['temperature']
checkpoint_every=CONFIG['checkpoint_every']
checkpoint_folder=CONFIG['checkpoint_folder']

# create checkpoint folder if it does not exist
if not checkpoint_folder:
    checkpoint_folder= './SFT_CustomAddition_Instruct_DIM'+str(decoder_dim)+'_RT'+str(lA)+'_SFT'+str(lSFT)+'_checkpoints'

print('checkpoint folder=',checkpoint_folder)

checkpoint_folder_path = Path(checkpoint_folder)
checkpoint_folder_path.mkdir(exist_ok = True, parents = True)

# save the config file in the checkpoint folder
config_save_path = checkpoint_folder+'/config.json'
with open(config_save_path, 'w') as fconfig:
    json.dump(CONFIG, fconfig)

# read the new config file 
with open(config_save_path, 'r') as fconfig:
    CONFIG = json.load(fconfig)


print('max_seq_len=',max_seq_len)
print('dataset_length=',dataset_length)
print('length_check=',length_check)
print('heads=',heads)
print('depth=',depth)
print('decoder_dim=',decoder_dim)
print('lSFT=',lSFT)
print('lA=',lA)
print('temperature=',temperature)

model_id=CONFIG['ref_model_id']

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
pad_token_id = tokenizer.pad_token_id
    
def tokenize_function(examples,dim):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=dim)

## create and load dataset 
    
def load_math(max_digit=CONFIG["max_digit"], dim=max_seq_len):
    random_number_1 = random.randint(0, max_digit)
    random_number_2 = random.randint(0, max_digit)
    q = 'Question: What is the sum of following two numbers '+str(random_number_1)+', '+str(random_number_2)+'?'
    a = 'Answer: '+ str(random_number_1+random_number_2)
    tokenized_qa = torch.tensor(tokenize_function(q+a,dim)['input_ids'])
    token_length_q = torch.sum(torch.tensor(tokenize_function(q,dim)['attention_mask']), dim=-1)                
    return (tokenized_qa, token_length_q)

features, labels = load_math()
print('Example question and answer from custom addition dataset: ',tokenizer.decode(features, skip_special_tokens=True))

sft_dataset_train = create_example_dataset(dataset_length, lambda: load_math())
sft_dataset_valid = create_example_dataset(int(dataset_length/2), lambda: load_math())

print("Loading dataset is successful.")

policy_model = TransformerWrapper(num_tokens = len(tokenizer),max_seq_len = 2*max_seq_len,attn_layers = Decoder(dim = decoder_dim,depth = depth,heads = heads))

# Count the total number of parameters
model = policy_model
total_params = sum(p.numel() for p in model.parameters())
# Count the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters in policy model: {total_params}")
print(f"Trainable parameters in policy model: {trainable_params}")

ref_model_id = model_id
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, device_map=device, torch_dtype=torch.float32,use_cache=False)

ref_model.gradient_checkpointing_enable()

print("Reference model is loaded successfully. aSFT training is starting.")

start_time = time.time()
Asft_trainer = aSFTTrainer(
    policy_model, 
    ref_model, 
    policy_model_from_HF=False,
    ref_model_from_HF=True,
    policy_model_id=None,
    ref_model_id=ref_model_id,
    train_sft_dataset = sft_dataset_train,
    valid_sft_dataset = sft_dataset_valid,
    valid_every = 1,
    data_type = 'tuple',
    pad_id = pad_token_id,
    max_seq_len = max_seq_len,
    batch_size = batchsize, 
    epochs = epochs,
    temperature = temperature,
    checkpoint_every = checkpoint_every,
    checkpoint_folder = checkpoint_folder,
    asft_kwargs = dict(
            λ = lSFT,
            λp = lA,
        ),
)

Asft_trainer()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

