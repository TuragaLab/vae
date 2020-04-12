import torch


class BetaReg():
    """Beta-distribution regularizer for parameters in range [0, 1].

    Example:
    # Instantiate callable regularizer:
    >>> beta_reg = BetaReg(0.1, 0.01, 0.001, gen_model.some_param, function=softplus)
    # Later in training loop
    >>> loss = some_loss(data)
    >>> loss = loss + beta_reg()
    >>> loss.backward()
    """

    def __init__(self, mean, std, lmbda, param, function=None):
        """
        Args:
            mean (float): Mean of beta distribution.
            sigma (float): Standard deviation of beta distribution.
            lmbda (float): Regularizer pre-factor.
            param (object): Parameter(s) to apply Beta to.
            function (callable): Optional function (e.g. softplus) to apply to param.
        """
        alpha = mean*(mean*(1-mean)/std**2-1)
        beta = alpha/mean * (1-mean)
        self.Beta = torch.distributions.beta.Beta(alpha, beta)
        self.lmbda = lmbda
        self.param = param
        if function is not None:
            self.function = function
        else:
            self.function = lambda x: x # identity

    def __call__(self):
        """Returns log-probability log Beta(param)."""
        return -self.lmbda*torch.sum(self.Beta.log_prob(self.function(self.param)))


class NormalReg():
    """Normal distribution regularizer.

    Example:
    # Instantiate callable regularizer:
    >>> normal_reg = NormalReg(0.1, 0.01, 0.001, gen_model.some_param, function=softplus)
    # Later in training loop
    >>> loss = some_loss(data)
    >>> loss = loss + normal_reg()
    >>> loss.backward()
    """

    def __init__(self, mean, std, lmbda, param, function=None):
        """
        Args:
            mean (float): Mean of Normal distribution.
            sigma (float): Standard deviation of Normal distribution.
            lmbda (float): Regularizer pre-factor.
            param (object): Parameter(s) to apply Normal to.
            function (callable): Optional function (e.g. softplus) to apply to param.
        """
        self.Normal = torch.distributions.normal.Normal(loc=mean, scale=std)
        self.lmbda = lmbda
        self.param = param
        if function is not None:
            self.function = function
        else:
            self.function = lambda x: x # identity

    def __call__(self):
        """Returns log-probability log Normal(param)."""
        return -self.lmbda*torch.sum(self.Normal.log_prob(self.function(self.param)))


def beta_regularizer(mean, std, lmbda, param, function=None):
    """Beta-distribution regularizer for parameters in range [0, 1].

    Functional version utilizing closure technique.

    Args:
        mean (float): Mean of beta distribution.
        std (float): Standard deviation of beta distribution.
        lmbda (float): Regularizer pre-factor.
        param (object): Parameter(s) to apply Beta to.
        function (callable): Optional function (e.g. softplus) to apply to param.

    Returns:
        beta_reg: Function to be called in training loop as `loss = vimco(data) + beta_reg()`

    Example:
    # Initialize callable regularizer:
    >>> beta_reg = beta_regularizer(0.1, 0.01, 0.001, gen_model.some_param, function=softplus)
    # Later in training loop
    >>> loss = some_loss(data)
    >>> loss = loss + beta_reg()
    >>> loss.backward()
    """
    alpha = mean * (mean * (1-mean)/std**2 - 1)
    beta = alpha/mean * (1-mean)
    if function is None:
        function = lambda x: x # identity
    def beta_reg():
        return -lmbda*torch.sum(torch.distributions.beta.Beta(alpha, beta).log_prob(function(param)))
    return beta_reg
