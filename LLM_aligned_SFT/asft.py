from pathlib import Path
import glob
import os
import pickle

from beartype import beartype
from beartype.typing import Optional, Callable, Union
from torchtyping import TensorType

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator

from einops import rearrange
from einx import get_at

from pytorch_custom_utils.utils import (
    masked_mean,
    maybe_and_mask
)

from pytorch_custom_utils.accelerate_utils import (
    model_forward_contexts
)

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

from torch.optim.lr_scheduler import LinearLR

from transformers import AutoTokenizer

from LLM_aligned_SFT.sampling_utils import (
    sample,
    top_p,
    top_k
)

from tqdm import tqdm

from ema_pytorch import EMA


def adam_optimizer_with_linear_decay(
    model: Module,
    start_learning_rate: float,
    end_learning_rate: float,
    num_decay_steps: int,
    accelerator: Accelerator,
    weight_decay: float,
    adam_kwargs: dict = dict(),
) -> OptimizerWithWarmupSchedule:

    adam = get_adam_optimizer(
        model.parameters(),
        lr = start_learning_rate,
        wd = weight_decay
    )

    scheduler = None
    if start_learning_rate != end_learning_rate:
        scheduler = LinearLR

    return OptimizerWithWarmupSchedule(
        optimizer = adam,
        accelerator = accelerator,
        scheduler = LinearLR,
        scheduler_kwargs = dict(
            start_factor = 1.,
            end_factor = end_learning_rate / start_learning_rate,
            total_iters = num_decay_steps
        )
    )



def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def log_prob_from_model_and_seq(model, seq, is_HF_model=False, policy_model_id=None, ref_model_id=None, model_type='policy', max_seq_len=128):
    
    # returns log probs of the generated sequence (of same length as padded seq) after passing seq through the model

    if is_HF_model:
        if policy_model_id is not None or ref_model_id is not None:
            if policy_model_id is not None:
                policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
                if policy_tokenizer.pad_token is None:
                    policy_tokenizer.pad_token = policy_tokenizer.eos_token
            if ref_model_id is not None:
                ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
                if ref_tokenizer.pad_token is None:
                    ref_tokenizer.pad_token = ref_tokenizer.eos_token
            if model_type=="ref":
                seq_decoded = [ref_tokenizer.decode(seq[i], skip_special_tokens=True) for i in range(seq.shape[0])]
                seq_encoded = ref_tokenizer(seq_decoded, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
                seq =  {k: torch.tensor(v).to("cuda") for k, v in seq_encoded.items()} 
            
    
        logits = model(**seq)['logits']
        log_probs = logits.log_softmax(dim = -1)
        return get_at('b n [c], b n -> b n', log_probs, seq['input_ids'])
    else:
        logits = model(seq)
        log_probs = logits.log_softmax(dim = -1)
        return get_at('b n [c], b n -> b n', log_probs, seq)


def prompt_mask_from_len(lengths, seq):

    # returns mask that ignores everything in seq after length

    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device = device) < rearrange(lengths, '... -> ... 1')

def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob


class aSFT(Module):
    def __init__(
        self,
        model: Module,
        ref_model: Union[Module, None], 
        *,
        policy_model_from_HF: bool,
        ref_model_from_HF: bool,
        policy_model_id: str,
        ref_model_id: str,
        max_seq_len: int=128,
        λ = 0.1,
        λp = 1,
        pad_id: Optional[int] = None,
        ref_model_ema_decay = 1.,
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.policy_model = model
        
        if ref_model is not None:
            self.ref_model = EMA(
                ref_model,
                beta = 0.0,
                **ema_kwargs
            )
        else:
            self.ref_model = EMA(
                model,
                beta = ref_model_ema_decay,
                **ema_kwargs
            )

        self.λ = λ
        self.λp = λp
        self.pad_id = pad_id
        self.policy_model_from_HF = policy_model_from_HF
        self.ref_model_from_HF = ref_model_from_HF
        self.policy_model_id = policy_model_id
        self.ref_model_id = ref_model_id
        self.max_seq_len = max_seq_len
        

    def update_reference_model_with_policy(self):
        self.ref_model.copy_params_from_model_to_ema()

    def update_ema(self):
        self.ref_model.update()

    def parameters(self):
        return self.policy_model.parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @autocast(enabled = False) 
    def forward(
        self,
        generated_seq: TensorType['b', 'n', int],
        real_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int],
        generated_seq_mask: Optional[TensorType['b', 'n', bool]] = None,
        real_seq_mask: Optional[TensorType['b', 'n', bool]] = None
    ):
        self.policy_model.train()

        real_prompt_mask = prompt_mask_from_len(prompt_len, real_seq) # mask to keep only question from real_seq
        generated_prompt_mask = prompt_mask_from_len(prompt_len, generated_seq) # mask to keep only question from generated_seq
        
        real_seq_prompt = torch.where(real_prompt_mask,real_seq, self.pad_id) # only the question with padding

        

        if exists(self.pad_id): # set padding to 0
            assert not exists(generated_seq_mask)
            assert not exists(real_seq_mask)
            generated_seq_mask = generated_seq != self.pad_id
            generated_seq.masked_fill_(~generated_seq_mask, 0)

            real_seq_mask = real_seq != self.pad_id
            real_seq.masked_fill_(~real_seq_mask, 0)
            
            real_seq_prompt_mask = real_seq_prompt != self.pad_id
            real_seq_prompt.masked_fill_(~real_seq_prompt_mask, 0)    

        with torch.no_grad():
            self.ref_model.eval()
            ref_generated_logprob = log_prob_from_model_and_seq(self.ref_model, generated_seq, is_HF_model=self.ref_model_from_HF, policy_model_id=self.policy_model_id, ref_model_id=self.ref_model_id, model_type='ref', max_seq_len=self.max_seq_len)
            ref_real_logprob = log_prob_from_model_and_seq(self.ref_model, real_seq_prompt, is_HF_model=self.ref_model_from_HF, policy_model_id=self.policy_model_id, ref_model_id=self.ref_model_id, model_type='ref', max_seq_len=self.max_seq_len)

        policy_generated_logprob = log_prob_from_model_and_seq(self.policy_model, generated_seq, is_HF_model=self.policy_model_from_HF, policy_model_id=self.policy_model_id, ref_model_id=self.ref_model_id, model_type='policy', max_seq_len=self.max_seq_len)
        policy_real_logprob = log_prob_from_model_and_seq(self.policy_model, real_seq_prompt, is_HF_model=self.policy_model_from_HF, policy_model_id=self.policy_model_id, ref_model_id=self.ref_model_id, model_type='policy', max_seq_len=self.max_seq_len)

        # masked mean for variable lengths

        policy_generated_logprob, ref_generated_logprob = [masked_mean(seq, maybe_and_mask(generated_seq_mask, ~generated_prompt_mask)) for seq in (policy_generated_logprob, ref_generated_logprob)]
        policy_real_logprob, ref_real_logprob = [masked_mean(seq, maybe_and_mask(real_seq_mask, ~real_prompt_mask)) for seq in (policy_real_logprob, ref_real_logprob)]

        # aSFT loss: first two terms corresponds to checking answering ability of the model when just question is passed, last two terms corresponds to checking correction/explanation when question and answer are passed 

        losses = -F.logsigmoid(self.λ * ((policy_real_logprob - ref_real_logprob) + self.λp * (policy_generated_logprob - ref_generated_logprob)))

        return losses.mean()


class aSFTTrainer(Module):
    def __init__(
        self,
        model: Union[Module, aSFT],
        ref_model: Union[Module, None], 
        *,
        policy_model_from_HF: bool,
        ref_model_from_HF: bool,
        policy_model_id: str,
        ref_model_id: str,
        train_sft_dataset: Dataset,
        data_type: str='tuple',
        max_seq_len: int,
        valid_sft_dataset: Optional[Dataset] = None,
        valid_every = 1,
        accelerator: Optional[Accelerator] = None,
        accelerate_kwargs: dict = dict(),
        batch_size = 18,
        grad_accum_steps = 2,
        epochs = 2,
        start_learning_rate = 1e-6,
        end_learning_rate = 1e-7,
        learning_rate_num_decay_steps = 1000,
        dropout = 0.,
        weight_decay = 0.,
        adam_kwargs: dict = dict(),
        temperature = 0.7,
        filter_fn = top_p,
        filter_kwargs = dict(thres = 0.9),
        pad_id: int = -1,
        ref_model_ema_decay = 1.,
        checkpoint_every = None,
        checkpoint_folder = './aSFT-checkpoints',
        asft_kwargs: dict = dict(
            λ = 0.1,
            λp = 1
        )
    ):
        super().__init__()

        self.accelerator = accelerator
        if not exists(self.accelerator):
            self.accelerator = Accelerator(**accelerate_kwargs)

        if not isinstance(model, aSFT):
            model = aSFT(
                model,
                ref_model, 
                policy_model_from_HF=policy_model_from_HF,
                ref_model_from_HF=ref_model_from_HF,
                policy_model_id=policy_model_id,
                ref_model_id=ref_model_id,
                max_seq_len=max_seq_len,
                pad_id = pad_id,
                ref_model_ema_decay = ref_model_ema_decay,
                **asft_kwargs
            )

        self.model = model
        self.policy_model_from_HF = policy_model_from_HF  
        self.ref_model_from_HF = ref_model_from_HF 
        self.policy_model_id = policy_model_id,
        self.ref_model_id = ref_model_id,
        self.max_seq_len = max_seq_len,
        self.dropout = dropout
        self.train_dataloader = DataLoader(train_sft_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        self.data_type = data_type
        
        self.grad_accum_steps = grad_accum_steps
        self.num_train_steps = len(self.train_dataloader) // self.grad_accum_steps * epochs

        self.optimizer = adam_optimizer_with_linear_decay(
            model,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps = learning_rate_num_decay_steps,
            accelerator = self.accelerator,
            weight_decay = weight_decay,
            adam_kwargs = adam_kwargs
        )
        
        # checkpointing

        self.should_checkpoint = exists(checkpoint_every)
        self.checkpoint_every = checkpoint_every

        if self.should_checkpoint:
            self.loss_path = os.path.join(checkpoint_folder,'loss_tracker.pkl')
            self.loss_path_valid = os.path.join(checkpoint_folder,'loss_tracker_validation.pkl')
            self.checkpoint_folder = Path(checkpoint_folder)
            self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        
        
        if glob.glob(str(self.checkpoint_folder)+'/*.pt'):
            checkpoint_path = glob.glob(str(self.checkpoint_folder)+'/*.pt')[0]
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model']
            self.model.load_state_dict(state_dict)
            self.base_ckpt_num = int(checkpoint_path.split('/')[-1].split('.')[2])
            print(f"Resumed training from previous checkpoint")
        else:
            
            self.base_ckpt_num = 0
        self.steps = 0  

        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # sampling

        self.temperature = temperature
        self.filter_fn = filter_fn
        self.filter_kwargs = filter_kwargs

        # validation

        self.valid_dataloader = None
        self.valid_every = valid_every

        if exists(valid_sft_dataset):
            self.valid_dataloader = DataLoader(valid_sft_dataset, batch_size = batch_size)
        
        if exists(valid_sft_dataset):
            (
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader,
                self.valid_dataloader
            )
            
        else:    
            (
                self.model,
                self.train_dataloader
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader
            )

    @property
    def is_main(self): 
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def print(self, *msg):
        self.accelerator.print(*msg)

    def log(self, **data):
        self.accelerator.log(data, step = self.steps)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def save(self, path: str, overwrite: bool = False):
        self.wait()

        if self.is_main:

            path = self.checkpoint_folder / path

            assert not path.exists() or overwrite, f'file already exists'

            pkg = dict(
                model = self.unwrapped_model.state_dict()
            )

            torch.save(pkg, str(path))

    def calc_asft_loss(
        self,
        real_seq: TensorType['b', 'n', int],
        prompt_len: TensorType['b', int],
        is_HF_model: bool
    ):
        prompt_mask = prompt_mask_from_len(prompt_len, real_seq)
        prompts = real_seq[prompt_mask].split(prompt_len.tolist())

        generated_seqs = sample(
            self.unwrapped_model.policy_model,
            prompts = prompts,
            seq_len = self.max_seq_len,
            temperature = self.temperature,
            filter_fn = self.filter_fn,
            filter_kwargs = self.filter_kwargs,
            output_keep_prompt = True, # generated sequence has the question and answer
        )

        asft_loss = self.model(
            real_seq = real_seq,
            generated_seq = generated_seqs,
            prompt_len = prompt_len
        )

        return asft_loss

    def forward(self, overwrite_checkpoints: bool = False):
        

        set_dropout_(self.model, self.dropout)

        train_dataloader_iter = cycle(self.train_dataloader)
        
        loss_tracker = []
        loss_tracker_valid = []

        for _ in tqdm(range(self.num_train_steps), desc = 'asft fine-tuning'):

            self.model.train()
            for forward_context in model_forward_contexts(self.accelerator, self.model, self.grad_accum_steps):
                with forward_context():
                    if self.data_type=='tuple':
                        real_seq, prompt_len = next(train_dataloader_iter) # real_seq = question + answer, prompt_len=len of question
                    elif self.data_type=='dict':
                        input_data = next(train_dataloader_iter)
                        real_seq = input_data['input_ids']
                        prompt_len = input_data['seq_len']

                    train_loss = self.calc_asft_loss(real_seq, prompt_len,self.policy_model_from_HF)
                    loss_tracker.append(train_loss)
                    with open(self.loss_path, 'wb') as file:  
                        pickle.dump(loss_tracker, file)

                    self.accelerator.backward(train_loss / self.grad_accum_steps)


            self.print(f'train asft loss: {train_loss.item():.3f}')
            self.log(loss = train_loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1

            self.wait()

            self.unwrapped_model.update_ema()

            if exists(self.valid_dataloader) and not (self.steps % self.valid_every):
                self.wait()

                if self.is_main:
                    total_loss = 0.
                    total_batches = 0.

                    with torch.no_grad():
                        self.model.eval()

                        for valid_seq, prompt_len in tqdm(self.valid_dataloader, desc = 'valid asft'):
                            batch = valid_seq.shape[0]
                            valid_asft_loss = self.calc_asft_loss(valid_seq, prompt_len,self.policy_model_from_HF)

                            total_batches += batch
                            total_loss += valid_asft_loss * batch

                        valid_loss = total_loss / total_batches
                        
                        loss_tracker_valid.append(valid_loss)
                        with open(self.loss_path_valid, 'wb') as file:  
                            pickle.dump(loss_tracker_valid, file)

                        self.print(f'valid asft loss: {valid_loss.item():.3f}')
                        self.log(valid_asft_loss = valid_loss.item())

            
            if self.should_checkpoint and self.steps%self.checkpoint_every==0:
                self.base_ckpt_num +=1 
                self.save(f'asft.ckpt.{self.base_ckpt_num}.pt', overwrite = overwrite_checkpoints)

        self.print(f'Aligned SFT training is completed.')