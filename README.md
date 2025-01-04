# knowledge-distillation-of-large-language-models

The standard approach of  [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347) to fine-tune LLMs learns a reward model from human preferences and then it uses it to train the LLM. Another alternative is to directly use human preferences to train the LLM known as [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290). In both cases, the approach is limited by the size and quality of the human-annotated data. When human-annotated data is limited, it is also common to use advanced LLMs to generate high-quality synthetic data to train a smaller language model. Another method of using a stronger LLM to improve the performance of a weaker one is to use the stronger one as a judge. In this repo, we ask a slightly different question:

When we have access to human-annotated high-quality data, is there any advantage to having access to generated responses of a stronger LLM on the same dataset?

To answer this question we set up a simple experiment in which we train a small 0.27 B transformer on [mathematics_dataset](https://arxiv.org/pdf/1904.01557) from Google, deepmind guided by [Llama 8B](https://ai.meta.com/blog/meta-llama-3/). For us, the loss function that we minimize at time t includes a contribution of Kullback–Leibler divergence between Llama and our model, more precisely for a given input prompt x,

$Loss(x,t) = \lambda( KL(pdata(|x), p(|x,t))-KL(pdata(|x), pLlama(|x))+\lambda'\int dy p(y|x,t) KL(p(|x,y,t),pLlama(|x,y)))$

Where there first two terms are the usual SFT loss function between the human-annotated probability distribution function pdata(|x) for the output and that of our model represented by p(|x,t), and the last term tries to push the performance of our model towards that of Llama for the feedback on generated prompt-response pairs. Note that this method does not require us to know the weights of the helping LLM (here Llama) as opposed to most knowledge-distillation techniques. The results are presented below:

<center>
<img alt="fig1" width="800px" src="LLM aligned SFT.png">
</center>

The experiment is performed on a transformer with 0.27B trainable parameters. From the plot, we see that as $\lambda'$ is increased the asymptotic value of the loss at large compute decreases significantly.  
