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

        self.mean_encoder = Encoder(
            input_dim, 
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim
        )
        self.logstd_encoder = Encoder(
            input_dim, 
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim
        )    

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
