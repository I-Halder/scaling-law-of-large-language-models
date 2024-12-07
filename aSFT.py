## Make all imports 
import torch
from torch import Tensor
import time
import sys
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


from huggingface_hub import login  
login(token="..") # make sure to add correct huggingface tokens 

start_time = time.time()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# define training parameters 
max_seq_len=128
dataset_length=100 
dim = 16
depth = 1
heads = 1
lambda_val = 0.1
lambda_p_val = 0.4
epochs = 1
temp = 0.01
checkpoint_folder = './checkpoints/checkpoints_lambdaP'+str(lambda_p_val)+'_lambda'+str(lambda_val)


print('max_seq_len=',max_seq_len)
print('dataset_length=',dataset_length)
print('Transformer dimension:', dim)
print('Number of heads:', heads)
print('Number of layers:', depth)
print('Lambda_p', lambda_p_val)
print('Lambda', lambda_val)
print('Number of epochs:', epochs)
print('Temperature:', temp)


model_id = "meta-llama/Llama-3.1-8B"   # model id for the helper LLM
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples,dim):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=dim)

dataset =load_dataset("deepmind/math_dataset", 'algebra__linear_1d',trust_remote_code=True) # add the dataset here
train_data=dataset['train']
test_data=dataset['test']

length_check=int(max_seq_len/2)
def load_deepmath(train_data, dim):
    random_index = random.randint(0, len(train_data) - 1)
    random_sample = train_data[random_index]
    q = random_sample['question']
    a = random_sample['answer']
    return (torch.tensor(tokenize_function(q+a,dim)['input_ids'],device=device),torch.tensor(min(len(q+a),length_check),device=device))

sft_dataset = create_example_dataset(dataset_length, lambda: load_deepmath(train_data, max_seq_len))

transformer = TransformerWrapper(
    num_tokens = len(tokenizer),
    max_seq_len = 2*max_seq_len,
    attn_layers = Decoder(
        dim = dim,
        depth = depth,
        heads = heads
    )
) # This is the model that is getting trained

helper_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.float16)
print('We are using '+model_id+ ' as the helper LLM.')

Asft_trainer = aSFTTrainer(
    transformer,
    helper_model,
    max_seq_len = max_seq_len,
    epochs= epochs,
    train_sft_dataset = sft_dataset,
    checkpoint_every = 1,
    checkpoint_folder = checkpoint_folder,
    temperature = temp,
    asft_kwargs = dict(
        λ = lambda_val,
        λp = lambda_p_val,
    )
)
Asft_trainer()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
