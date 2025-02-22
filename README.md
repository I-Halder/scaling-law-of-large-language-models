<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;E=mc^2" />
</td></tr>
</table>

# Review of reinforcement learning from human feedback
We are presented with a prompt $x$ and a set of $K$ answers with ground truth preference $y_1> \ldots>y_K$. The language model generates response $y$ form $\pi_\theta(y|x)$. We define penalty/error  $E(y|x)$ for response $y$ using the following formula
<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;min_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(y|x)}[E(y|x)+T\mathbb{D}_{\textrm{KL}}[\pi_{\theta}(y|x)||\pi_{ref}(y|x)]]" />
</td></tr>
</table>

Given a permutation $\tau$, the Plackett-Luce model assigns a probability distribution for the generated response of the language model
<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;p(y_{\tau(1)}>\ldots>y_{\tau(K)}|x)=\prod_{k=1}^{K}\frac{\exp(-E(y_{\tau(k)}|x))}{\sum_{j=k}^{K}\exp(-E(y_{\tau(j)}|x))}" />
</td></tr>
</table>

In usual approach to reinforcement learning we minimize the following objective 
<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;-min_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y_1>y_2>\dots>y_K}\bigl[log\hspace{0.1cm}p(y_1>y_2>\dots>y_K|x)\bigr]" />
</td></tr>
</table>

In Proximal Policy Optimization (PPO) one solves the last equation to get $E$ and then finds the optimal model from the first equation. On the other hand, for Direct Preference Optimization (DPO), first one finds $E$ in terms of $\pi_\theta$ from the first equation and then solves the last equation to find optimal $\pi_\theta$. This leads to the following effective objective
<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_{\text{DPO}}(\pi_{\theta})=-\mathbb{E}_{x\sim\mathcal{D},y_1>\dots>y_K}[log\prod_{k=1}^{K}\frac{e^{T\hspace{0.1cm}log\frac{\pi_{\theta(y_{k}|x)}}{\pi_{ref}(y_{k}|x)}}}{\sum_{j=k}^{K}e^{T\hspace{0.1cm}log\frac{\pi_{\theta}(y_{k}|x)}{\pi_{ref}(y_{k}|x)}}}]" />
</td></tr>
</table>
