# knowledge-distillation-of-large-language-models

The standard approach of  [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347) to fine-tune LLMs learns a reward model from human preferences  and then it uses it to train the LLM. Another alternative is to directly use human preferences to train the LLM known as [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290). In both cases, the approach is limited by the size and quality of the human-annotated data. When human-annotated data is limited, it is also common to use advanced LLMs to generate high-quality synthetic data to train a smaller language model. Another method of using a stronger LLM to improve performace of a weaker one is to use the stronger one as judge. In this repo, we ask a slightly different question:

When we have access to human-annotated high-quality data, is there any advantage to having access to generated responses of a stronger LLM on the same dataset?

<center>
<img alt="fig1" width="800px" src="LLM aligned SFT.png">
</center>
