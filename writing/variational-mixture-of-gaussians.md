# General Approach

Typical Normal 

$$
p(z) ~ N(0, I)
$$

Multivariate Normal

$$
p(z) ~ \Sum_{k=1}^{K} \pi_k * N(z; \mu_k, \sigma_k^2I)
$$

## Estimation of probability
```{python}

# We define the parameters for the centers.
means = nn.Parameter(torch.zeros(n_centers, 10))

# We define the parameters for the std deviations.
logstd = nn.Parameter(torch.rand(n_centers, 10))

# We define the parameters for their weights.
weights = nn.Parameter(torch.zeros(n_centers))

def mixture_log_probability(z):
    log_probability = []
    for k in range(n_classes):
        mu_k = means[k]
        std_k = logstd[k].exp()
        log_pk = Normal(mu_k, std_k).log_prob(z).sum(axis=1)
        log_probability(log_pk + F.log_softmax(mixture_logits, dim=0)[k])
    log_probs = torch.stack(log_probs, dim=0)
    log_mix = torch.logsumexp(log_probs, dim=0)
    return log_mix
```

## Numerically Stable Version.

```{python}
def mixture_log_prob_stable(z, mixture_means, mixture_log_std, mixture_logits):
    normal_log_probs = -0.5 * (
        ((z_expanded - means) / stds)**2 + \
        2 * mixture_log_stds + np.log(2*np.pi)
    )
```
