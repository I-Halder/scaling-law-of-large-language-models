## Description of the code

This repo contains significantly modified version of the code in https://github.com/lucidrains/self-rewarding-lm-pytorch. 

## Scaling law of large language models

We train a transformer with single heads of head_dim 64, max_seq_len 256, vocab_dim 128256, feed_forward ratio 4 from scratch for various values of hidden_dim. We use a simple addition dataset with 1000 points as the training set. In the plot below we compare the validation loss againt the computational cost in FLOP. The FLOP is calculated after excluding the embedding and logit layers.

<div style="display: flex; justify-content: center;">
    <img alt="fig1" width="1000px" src="loss_vs_compute.jpg" style="transform: translateX(30px);">
</div>

From the plot we see that given a computational budget, there is an optimal model size.

