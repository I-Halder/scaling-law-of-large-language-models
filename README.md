# knowledge-distillation-of-large-language-models


It is common to use advanced LLMs to generate high-quality synthetic data to fine tune smaller language models. Also, recently there has been a lot of interest in using a strong LLM as judge to evaluate performace of a weaker model. For the training of a weaker language model, we ask the following question: 

When we have access to human annotated high quality data, is there any advantage to having access to  high-quality generated response of a stronger LLM on the same dataset?

<center>
<img alt="fig1" width="800px" src="LLM aligned SFT.png">
</center>
