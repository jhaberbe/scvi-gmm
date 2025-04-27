# Basics 

The generative model used is that:

$$
z_n ~ Normal(0, I)
$$

low dimensional representation in some multivariate normal space.

$$
x_ng ~ NB(l_n * f^g(z_n), \theta_g)
$$

so our detected output for feature g in sample n is some function of our total UMIs, multiplied by some function (the neural network), that maps the latent dimensional space to our output space (ie, the predicted mean of the cell), we also assume in the context of this model that we learn the dispersion parameter underlying the negative binomial distribution.  

# Base neural network to be used.

A very simple neural network.

```{python}
import torch
from torch import nn

class MyNeuralNetwork(nn.Module):

    def __init__(self, n_input: int, n_output: int, link_var = None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128)
            nn.ReLU(),
            nn.Linear(128, n_output)
        )

        if link_var == None:
            self.transformation = None
        if link_var == "softmax":
            self.transformation = nn.Softmax(dim=-1)
        if link_var == "exp":
            self.transformation = torch.exp
        else:
            self.transformation = link_var

    def forward(self, x: torch.Tensor):
        output = self.model(x)
        if self.transformation:
            return self.transformation(output)
        else:
            None
```

# Crafting in vanilla PyTorch

```{python}
class MyModule(BaseModuleClass):
    def __init__(self, n_input: int, n_latent: int = 10):
        super().__init__()

        # Setup our decoder module
        self.decoder = MyNeuralNetwork(n_latent, n_input, "softmax")
        # Log 
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))
        
        # Setup the parameters of the variational distribution
        # Still don't really understand wtf this means.
        # means are normal, and we encode the logvar
        self.mean_encoder = MyNeuralNetwork(n_input, n_latent, "none")
        self.var_encoder = MyNeuralNetwork(n_input, n_latent, "none")

    def inference(self, x: torch.Tensor):
        x_ = torch.log1p(x)
        # We learn the means and the variances of the underlying encoder.
        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)
        
        # We literally draw from the underlying distribution, so we get an 
        # understanding of if the noise we're predicting is correct or not.
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        # We'll return all the values out.
        return {
            "qz_m": qz_m,
            "qz_v": qz_v,
            "z": z,
        }
    
    def _get_generative_input(self, X: torch.Tensor, inf_out: dict[str, torch.Tensor]):

        # We want the latent dimension predicted (or really sampled) and get 
        # the library size (size factors) for each one.
        return {
            "z": inf_out["z"],
            "library": torch.sum(X, axis=1, keepdim=True)
        }

    def generative(self, z: torch.tensor, library: torch.tensor):
        # Return means for the negative binomial model.
        px_scale = self.decoder(z)

        # compute means
        px_mean = library * px_scale

        # dispersion
        theta = torch.exp(self.log_theta)

        return {
            "px_scale": px_scale,
            "px_mean": px_mean,
            "theta": self.log_theta,
        }
    
    def loss(self, tensors, inference_outputs, generative_outputs):
        x = tensors["x"]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()

        # Term 1
        log_lik = NegativeBinomial(total_count = theta, logits = nb_logits) \
            .log_prob(x) \
            .sum(dim=-1)
       
        # Term 2
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(axis=1)

        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)
```
