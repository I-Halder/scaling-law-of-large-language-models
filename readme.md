# Review of reinforcement learning from human feedback
We are presented with a prompt $x$ and a set of $K$ answers with ground truth preference $y_1> \ldots>y_K$. The language model generates response $y$ form $\pi_\theta(y|x)$. We define penalty/error  $E(y|x)$ for response $y$ using the following formula
\begin{equation}
min_{\pi_{\theta}}  \mathbb{E}_{x\sim \mathcal{D}, y\sim \pi_{\theta}(y \mid x)} \bigl[E( y|x) + T\mathbb{D}_{\textrm{KL}}\bigl[\pi_{\theta}(y\mid x)\mid \mid \pi_{ref}(y\mid x)\bigr]\bigr]
\end{equation}
Given a permutation $\tau$, the Plackett-Luce model assigns a probability distribution for the generated response of the language model
\begin{equation}
    p(y_{\tau(1)}>\ldots> y_{\tau(K)}| x)= \prod_{k=1}^{K}\frac{\exp(-E( y_{\tau(k)}|x))}{\sum_{j=k}^{K}\exp(-E( y_{\tau(j)}|x))}
\end{equation}
In usual approach to reinforcement learning we minimize the following objective 
\begin{equation}
-\min_{\pi_{\theta}}  \mathbb{E}_{x\sim \mathcal{D}, y_1>y_2>\dots>y_K}\bigl[\log p(y_1>y_2>\dots>y_K|x)\bigr]
\end{equation}
In Proximal Policy Optimization (PPO)  one solves (\ref{eq:RL2}) to get $E$ and then finds the optimal model from (\ref{eq:RL1}). On the other hand, for Direct Preference Optimization (DPO), first one finds $E$ in terms of $\pi_\theta$ from (\ref{eq:RL1}) and then solves (\ref{eq:RL2})  to find optimal $\pi_\theta$. This leads to the following effective objective 
\begin{equation}
    \mathcal{L}_{\text{DPO}}(\pi_{\theta}) = -\mathbb{E}_{ x\sim\mathcal{D}, y_1> \ldots> y_K}\left[\log \prod_{k=1}^{K}\frac{\exp\left(T \log \frac{\pi_{\theta}(y_{k}|x)}{\pi_{ref}(y_{k}|x)}\right)}{\sum_{j=k}^{K}\exp\left(T \log \frac{\pi_{\theta}(y_{k}|x)}{\pi_{ref}(y_{k}|x)}\right)}\right]
\end{equation}
## Self-Play Fine-Tuning

Self-Play Fine-Tuning (SPIN) corresponds to $K=2$ \cite{chen2024selfplayfinetuningconvertsweak}. In this case we set $\pi_{ref}=\pi_{\theta(t)}, y_1\sim \pi_{data}(y_1|x), y_2\sim\pi_{\theta(t)}(y_2|x)$.
\begin{equation}
    \mathcal{L}_{\text{DPO}}(\pi_{\theta}) = -\mathbb{E}_{ x\sim\mathcal{D}, y_1\sim \pi_{data}(y_1|x), y_2\sim\pi_{\theta(t)}(y_2|x), y_1>y_2}\left[\log \frac{\exp\left(T \log \frac{\pi_{\theta}(y_{1}|x)}{\pi_{\theta(t)}(y_{1}|x)}\right)}{\sum_{j=1}^{2}\exp\left(T \log \frac{\pi_{\theta}(y_{k}|x)}{\pi_{\theta(t)}(y_{k}|x)}\right)}\right]
\end{equation}

## Aligned Supervised Fine Tuning

 Aligned Supervised Fine Tuning (aSFT) corresponds to $K=3$. In this case we set $\pi_{ref}=\pi_{\theta(t)}, y_1\sim \pi_{data}(y_1|x), y_2\sim\pi_{LLM}(y_2|x), y_3\sim\pi_{\theta(t)}(y_3|x)$. Here we are using a larger languge model for assistance in generating $y_2$. 
\begin{equation}
    \mathcal{L}_{\text{DPO}}(\pi_{\theta}) = -\mathbb{E}_{ x\sim\mathcal{D}, y_1> \ldots> y_K}\left[\log \prod_{k=1}^{K}\frac{\exp\left(T \log \frac{\pi_{\theta}(y_{k}|x)}{\pi_{ref}(y_{k}|x)}\right)}{\sum_{j=k}^{K}\exp\left(T \log \frac{\pi_{\theta}(y_{k}|x)}{\pi_{ref}(y_{k}|x)}\right)}\right]
\end{equation}


