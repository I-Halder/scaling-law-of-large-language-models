<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;E=mc^2" />
</td></tr>
</table>

We are presented with a prompt $x$ and a set of $K$ answers with ground truth preference $y_1> \ldots>y_K$. The language model generates response $y$ form $\pi_\theta(y|x)$. We define penalty/error  $E(y|x)$ for response $y$ using the following formula
<table align="center">
<tr><td>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;min_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(y|x)}[E(y|x)+T\mathbb{D}_{\textrm{KL}}[\pi_{\theta}(y|x)||\pi_{ref}(y|x)]]" />
</td></tr>
</table>
