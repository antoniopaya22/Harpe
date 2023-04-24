# =================================================================
#
#                       WGAN-GP Utils
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
import numpy as np
import torch as th
import torch.autograd as autograd

from torch.autograd import Variable as V

# ==================> Functions
def compute_gradient_penalty(D: th.nn.Module, normal_t: th.Tensor, attack_t: th.Tensor)-> th.Tensor:
    """
    Computes gradient penalty loss for WGAN GP.

    :param D: th.nn.Module, discriminator.
    :param normal_t: th.Tensor, normal data.
    :param attack_t: th.Tensor, attack data.
    :return: th.Tensor, gradient penalty loss.
    """
    alpha = th.Tensor(np.random.random((normal_t.shape[0], 1)))
    between_n_a = (alpha * normal_t + ((1 - alpha) * attack_t)).requires_grad_(True)
    d_between_n_a = D(between_n_a)
    adv = V(th.Tensor(normal_t.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=d_between_n_a,
        inputs=between_n_a,
        grad_outputs=adv,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def create_batch(x: list, batch_size: int) -> list:
    """
    Create a suffled batch.

    :param x: list, list of data.
    :param batch_size: int, batch size.
    :return: list, list of batches.
    """
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    batch_x = [x[batch_size * i : (i+1)*batch_size,:] for i in range(len(x)//batch_size)]
    return batch_x