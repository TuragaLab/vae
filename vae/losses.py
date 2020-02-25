"""A set of model-agnostic losses implementing uniform API.

Implemented losses:

    VIMCO loss
    https://arxiv.org/abs/1602.06725

    All three Reweighted Wake-Sleep losses: PWake, QWake, QSleep,
    https://arxiv.org/abs/1406.2751

inf_net, gen_model, train_data must comply with the following interface:

    data = [x1, x2, ...] # list of observed data variables
    latents = [z1, z2, ...] # list of latent variables

    Recognition model interface:
        inf_net(data) # forward pass, updates internal state
        some_param = inf_net.some_param_ # query for distribution params
        latents = inf_net.sample() # sample z ~ q(z|data)
        log_q = inf_net.log_prob(latents)

    Generative model interface:
        gen_model(data, latents) # forward pass, updates internal state
        some_param = gen_model.some_param_ # query for distribution params
        log_p = gen_model.log_prob(data, latents)
        samples = gen_model.sample() # sample x, z ~ p(data, latents)

    log_q and lol_p shapes must be (batch_size, n_samples).
"""

import torch


class VIMCO(torch.nn.Module):
    """Model-agnostic vectorized VIMCO loss for both rec and gen model updates.

    VIMCO paper: https://arxiv.org/abs/1602.06725

    Attributes:
        loss_history_ (list): List of scalar losses. Appended at every iteration.
        mean_log_p_ (list): List of log_p averaged over data points (batch) and samples.
            Appended at every training iteration.
        mean_log_q_ (list): Same as mean_log_p, but for log_q.

    Methods:
        Call to instance: Performs forward pass and returns scalar VIMCO loss.
    """

    def __init__(self, inf_net, gen_model):
        """
        Args:
            inf_net (object): Recognition model instance.
            gen_model (object): Generative model instance.

        inf_net & gen_model instances must comply with the interface outlined above.
        """
        super().__init__()
        self.inf_net = inf_net
        self.gen_model = gen_model
        self.loss_history_ = [] # To be appended at each training iteration
        self.mean_log_p_ = []
        self.mean_log_q_ = []

    def forward(self, data):
        """Forward pass for VIMCO loss.

        Args:
            data (list): Train batch. A list of observed variables, [x1, x2, ...].

        Returns:
            loss (tensor): Scalar VIMCO loss.

        Updates attributes:
            loss_history_ (list of scalars)
            mean_loss_p_ (list of scalars)
            mean_loss_q_ (list of scalars)
        """
        # Step 1: Forward pass
        latent_samples = self.inf_net(data).sample() # forward pass and sample
        log_q = self.inf_net.log_prob(latent_samples) # (batch_size, n_samples)
        log_p = self.gen_model(data, latent_samples).log_prob(data, latent_samples) # (batch_size, n_samples)
        # Step 2: Calculate VIMCO loss
        batch_size, n_samples = log_q.shape
        K = torch.tensor(n_samples)
        log_f = log_p - log_q
        log_fh = (torch.sum(log_f, dim=1).unsqueeze(1)-log_f) / (K-1)
        Log_f = (torch.ones(batch_size, K, K) - torch.eye(K).unsqueeze(0)) * log_f.unsqueeze(1) + torch.diag_embed(log_fh)
        L = torch.logsumexp(log_f, dim=1) - torch.log(K.float())
        Li = torch.logsumexp(Log_f, dim=2) - torch.log(K.float())
        w = torch.nn.functional.softmax(log_f, dim=1)
        objective = w.detach()*log_f + (L.unsqueeze(1)-Li).detach()*log_q # (batch_size, n_samples)
        loss = -torch.mean(objective) # scalar loss
        # Step 3: Update class attributes
        self.loss_history_.append(loss.item())
        self.mean_log_q_.append(log_q.mean().item())
        self.mean_log_p_.append(log_p.mean().item())
        return loss


class PWake(torch.nn.Module):
    """Model-anostic wake phase loss for generative model update (p-wake loss).

    This is a part of the reweighted wake-sleep (RWS) training algorithm.
    RWS paper: https://arxiv.org/abs/1406.2751

    Attributes:
        loss_history_ (list): List of scalar losses. Appended at every iteration.
        mean_log_p_ (list): List of log_p averaged over data points (batch) and samples.
            Appended at every training iteration.
        mean_log_q_ (list): Same as mean_log_p, but for log_q.

    Methods:
        Call to instance: Performs forward pass and returns scalar p-wake loss.
    """

    def __init__(self, inf_net, gen_model):
        """
        Args:
            inf_net (object): Recognition model instance.
            gen_model (object): Generative model instance.
        """
        super().__init__()
        self.inf_net = inf_net
        self.gen_model = gen_model
        self.loss_history_ = [] # To be appended at each training iteration
        self.mean_log_p_ = [] # etc..
        self.mean_log_q_ = []

    def forward(self, data):
        """Forward pass for p-wake RWS loss.

        Args:
            data (list): Train batch. A list of observed variables, [x1, x2, ...].

        Returns:
            loss (tensor): Scalar VIMCO loss.

        Updates attributes:
            loss_history_ (list of scalars)
            mean_loss_p_ (list of scalars)
            mean_loss_q_ (list of scalars)
        """
        # Step 1: Forward pass
        latent_samples = self.inf_net(data).sample() # forward pass and sample
        log_q = self.inf_net.log_prob(latent_samples) # (batch_size, n_samples)
        log_p = self.gen_model(data, latent_samples).log_prob(data, latent_samples) # (batch_size, n_samples)
        # Step 2: Loss calculation
        log_w = (log_p - log_q).detach() # (batch_size, n_samples)
        w_norm = torch.nn.functional.softmax(log_w, dim=1) # (batch_size, n_samples)
        wake_p_loss = -torch.sum(w_norm*log_p) # scalar loss
        # Step 3: Update class attributes
        self.loss_history_.append(wake_p_loss.item())
        self.mean_log_q_.append(log_q.mean().item())
        self.mean_log_p_.append(log_p.mean().item())
        return wake_p_loss


class QWake(torch.nn.Module):
    """Model-anostic wake phase loss for inference network update (q-wake loss).

    This is a part of the reweighted wake-sleep (RWS) training algorithm.
    RWS paper: https://arxiv.org/abs/1406.2751

    Attributes:
        loss_history_ (list): List of scalar losses. Appended at every iteration.
        mean_log_p_ (list): List of log_p averaged over data points (batch) and samples.
            Appended at every training iteration.
        mean_log_q_ (list): Same as mean_log_p, but for log_q.

    Methods:
        Call to instance: Performs forward pass and returns scalar p-wake loss.
    """

    def __init__(self, inf_net, gen_model):
        """
        Args:
            inf_net (object): Recognition model instance.
            gen_model (object): Generative model instance.
        """
        super().__init__()
        self.inf_net = inf_net
        self.gen_model = gen_model
        self.loss_history_ = [] # To be appended at each training iteration
        self.mean_log_p_ = [] # etc..
        self.mean_log_q_ = []

    def forward(self, data):
        # Step 1: Forward pass
        latent_samples = self.inf_net(data).sample() # forward pass and sample
        log_q = self.inf_net.log_prob(latent_samples) # (batch_size, n_samples)
        log_p = self.gen_model(data, latent_samples).log_prob(data, latent_samples) # (batch_size, n_samples)
        # Step 2: Loss calculation
        log_w = (log_p - log_q).detach() # (batch_size, n_samples)
        w_norm = torch.nn.functional.softmax(log_w, dim=1) # (batch_size, n_samples)
        wake_q_loss = -torch.sum(w_norm*log_q) # scalar loss
        # Step 3: Update class attributes
        self.loss_history_.append(wake_p_loss.item())
        self.mean_log_q_.append(log_q.mean().item())
        self.mean_log_p_.append(log_p.mean().item())
        return wake_q_loss


class QSleep(torch.nn.Module):
    """Model-agnostic sleep phase loss for inference network update (q-sleep loss).

    This is a part of the reweighted wake-sleep (RWS) training algorithm.
    RWS paper: https://arxiv.org/abs/1406.2751

    Attributes:
        loss_history_ (list): List of scalar losses. Appended at every iteration.

    Methods:
        Call to instance: Performs forward pass and returns scalar q-sleep loss.
    """

    def __init__(self, inf_net, gen_model):
        """
        Args:
            inf_net (object): Recognition model instance.
            gen_model (object): Generative model instance.

        inf_net & gen_model instances must comply with the interface outlined above.
        """
        super().__init__()
        self.inf_net = inf_net
        self.gen_model = gen_model
        self.loss_history_ = [] # To be appended at each training iteration

    def forward(self, data):
        """Forward pass for QSleep update.

        Gets dream samples (data and latents) from generative model,
        feeds data to inference network, and calculates log prob of dream latents.

        Args:
            data (list): A list of observed variables, [x1, x2, ...].
                Not used in this sleep loss, but must be passed
                to comply with interface.

        Returns:
            sleep_q_loss (tensor): Scalar sleep loss.
        """
        dream_data, dream_latents = self.gen_model.sample()
        log_q = self.inf_net(dream_data).log_prob(dream_latents) # (batch_size, n_samples)
        sleep_q_loss = -torch.sum(log_q) # scalar loss
        self.loss_history_.append(sleep_q_loss.item())
        return sleep_q_loss
