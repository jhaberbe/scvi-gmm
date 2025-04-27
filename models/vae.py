import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, NegativeBinomial, Categorical, MixtureSameFamily, Independent

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MessageSender(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(MessageSender, self).__init__()

        self.mean_encoder = Encoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.logstd_encoder = Encoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    def forward(self, x):
        # Is this necessary (comes from scVI basic doumentation).
        log_x = torch.log1p(X)
        mu = self.mean_encoder(x)
        logstd = self.logstd_encoder(x)
        return mu, logstd

    def sample(self, x):
        mu, logstd = self.forward(x)
        return Normal(mu, torch.exp(logstd)).sample()

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MessageReceiver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(MessageReceiver, self).__init__()

        self.alpha_decoder = Decoder(X.shape[1])
        self.beta_decoder = Decoder(X.shape[1])

    def forward(self, z):
        alpha = self.alpha_decoder(z)
        beta = self.beta_decoder(z)
        return F.softplus(alpha), beta

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(VAE, self).__init__()
        # Message Passing
        self.message_sender = MessageSender(X.shape[1])
        self.message_receiver = MessageReceiver(X.shape[1])

        # Latent Representation Regularization
        self.n_gaussians = 10
        self.weights = nn.Parameter(torch.zeros(self.n_gaussians)).to("cuda")
        self.means = nn.Parameter(torch.zeros(self.n_gaussians, 64)).to("cuda")
        self.logstds = nn.Parameter(torch.zeros(self.n_gaussians, 64)).to("cuda")

    def forward(self, x):
        z = self.message_sender.sample(x)
        alpha, beta = self.message_receiver(z)
        return z, alpha, beta
    
    def loss(self, x):
        z, alpha, beta = self.forward(x)
        beta += (x.sum(axis=1, keepdims=True) / x.sum(axis=1, keepdims=True).mean()).log()
        nb_nll = -NegativeBinomial(total_count=alpha, logits=beta).log_prob(X).sum(axis=1)
        mixing_distribution = Categorical(probs=torch.softmax(self.weights, dim=0))
        component_distribution = Independent(Normal(self.means, self.logstds.exp() + 1e-4), 1)
        gmm = MixtureSameFamily(mixing_distribution, component_distribution)
        latent_nll = -gmm.log_prob(z)

        return (latent_nll + nb_nll).mean()
