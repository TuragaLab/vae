vae
===

Model-agnostic variational autoencoder tools.

Here we are putting together a set of generic modules for variational autoencoder related tasks.

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

Important distinction between discrete and continuous latent variables...

Add abstract classes (templates)...

Wake update sequence...

Sleep update sequence...

Short version with one-liners...

Concrete examples with, say, logits for spike-inference...
