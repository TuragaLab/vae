vae
===

Model-agnostic variational autoencoder tools.

Here we are putting together a set of generic modules for variational autoencoder related tasks.

In what follows, no assumptions are made as to the nature of latent variables (they can be both discrete and continuous).
However, it *is* assumed that the model is a "black-box" one, i.e. we consider the most general case without resorting to reparametrization.
Special cases allowing reparametrization will be considered/implemented later.

Uniform variational autoencoder interface
=========================================

We follow a uniform API for inference network, generative model, and loss
calculation, allowing for modularity and ease of use. The API is as follows.

The **inference network** and **generative model** essentially represent probability
distributions and therefore have the same API implementing  ``forward``, ``sample``,
and ``log_prob`` methods. Conceptually, this amounts to:

1. Calling the object. This will perform forward pass and update the internal state of the underlying probability distribution.
2. Sampling from the probability distribution.
3. Log-probability calculation.

The **loss** API amounts to simply passing the entire model and training batch
to the loss object. This will allow the loss object to drive the model (using
the model's API described above) in accordance with its specific forward pass
requirements (e.g., the wake update forward pass is different from sleep
update's one). The loss instance will also collect and store loss history during training.

In more detail, the API is:

:Inference network:

 ``inf_net = InfNet(hyperparameters)`` # instantiate inference network, pass all
 required hyperparameters

 ``inf_net(data)`` # forward pass, updates some ``params_`` attributes of
 q(z|x), returns ``self``. Conceptually, this is the "given-x" part of q(z|x).

 ``params = inf_net.params_`` # attribute query

 ``latents = inf_net.sample()`` # returns latent samples from the
 distribution, z ~ q(z|x)

 ``log_q = inf_net.log_prob(latents)`` # calculates posterior log-probability,
 log q(z|x)

:Generative model:

 ``gen_model = GenModel(hyperparameters)`` # instantiate generative model, pass
 all required hyperparameters

 ``gen_model(data, latents)`` # forward pass, updates ``params_`` attributes of
 p(x,z), returns self

 ``params_ = gen_model.params_`` # attribute query

 ``x_samples, z_samples = gen_model.sample()`` # returns samples x, z ~ p(x,z)

 ``log_p = gen_model.log_prob(data, latents)`` # calculates joint
 log-probability, log p(x,z)

:Loss:

 ``some_loss = SomeLoss(inf_net, gen_model)`` # instantiate loss object, pass
 inference network and generative model instances

 ``loss = some_loss(data)`` # performs forward pass and returns scalar loss to perform backward pass on

 The loss instance ``some_loss`` also collects and stores training history losses, which can be queried for:

 ``loss_history = some_loss.loss_history_``

 Also, optionally, loss object collects additional information, such as ``log_p`` and ``log_q`` for VIMCO loss, for example.

:Data format:

 The shape and format of ``data``, ``latents`` and ``samples`` must be consistent with the model's ``inf_net`` and ``gen_model`` expected input and output, which is up to the user. On the other hand, the ``forward`` method of the ``loss`` object expects ``log_q`` and ``log_p`` to be 2D tensors with shape [batch_size, n_samples].


**TODO:**

Add abstract classes (templates)...

Wake update sequence...

Sleep update sequence...

Short version with one-liners...

Concrete examples with, say, logits for spike-inference...

References
==========

General VAE
-----------
- Diederik P. Kingma and Max Welling, Auto-Encoding Variational Bayes, arXiv:1312.6114v10 [stat.ML], 2013.
- Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra, Stochastic Backpropagation and Approximate Inference in Deep Generative Models, arXiv:1401.4082 [stat.ML], 2014.
- Yuri Burda, Roger Grosse, Ruslan Salakhutdinov, Importance Weighted Autoencoders, arXiv:1509.00519v4 [cs.LG], 2015.

Gradient estimation, general
----------------------------
- John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel, Gradient Estimation Using Stochastic Computation Graphs, arXiv:1506.05254v3 [cs.LG], 2016.
- Shakir Mohamed, Mihaela Rosca, Michael Figurnov, Andriy Mnih, Monte Carlo Gradient Estimation in Machine Learning, arXiv:1906.10652v1 [stat.ML], 2019.

Optimization algorithms and gradient estimation
-----------------------------------------------
- Andriy Mnih and Danilo J. Rezende, Variational Inference for Monte Carlo Objectives, arXiv:1602.06725v2 [cs.LG], 2016.
- Andriy Mnih and Karol Gregor, Neural Variational Inference and Learning in Belief Networks, arXiv:1402.0030v2 [cs.LG], 2014.
- Jorg Bornschein, Yoshua Bengio, Reweighted Wake-Sleep, arXiv:1406.2751v4 [cs.LG], 2015.
- Tuan Anh Le, Adam R. Kosiorek, N. Siddharth, Yee Whye Teh, Frank Wood, Revisiting Reweighted Wake-Sleep, arXiv:1805.10469v2 [stat.ML], 2019.
- Chris J. Maddison, Andriy Mnih, Yee Whye Teh, The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables, arXiv:1611.00712v3 [cs.LG], 2017.
- Eric Jang, Shixiang Gu, Ben Poole, Categorical Reparameterization with Gumbel-Softmax, arXiv:1611.01144v5 [stat.ML], 2017.
