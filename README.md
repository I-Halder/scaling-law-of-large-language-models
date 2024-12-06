# knowledge-distillation-of-large-language-models

The standard approach of reinforcement learning to fine-tune LLMs learns a reward model from human preferences. The reward model is then used to train the LLM known as [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347). Another alternative is to directly use human preferences to train the LLM known as [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290). In both cases, the approach is limited by the size and quality of the human-annotated data. When human-annotated data is limited, it is also common to use advanced LLMs to generate high-quality synthetic data to train a smaller language model. Recently, there has been a lot of interest in using a strong LLM as a judge to evaluate the performance of a weaker model and thereby improve it. For the training of a weaker language model, we ask the following question: 

When we have access to human-annotated high-quality data, is there any advantage to having access to the high-quality generated response of a stronger LLM on the same dataset?

<center>
<img alt="fig1" width="800px" src="LLM aligned SFT.png">
</center>
