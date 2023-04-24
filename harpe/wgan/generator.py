# =================================================================
#
#                       WGAN-GP Generator
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
import torch as th
import numpy as np

from torch import nn


# ==================> Class
class Generator(nn.Module):
    """
    Generator class for WGAN-GP.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Constructor.

        :param input_dim: int, dimension of the latent space.
        :param output_dim: int, dimension of the output space.
        """
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, noise_dim, raw_attack, nonfunctional_features_index):
        batch_size = len(raw_attack)
        noise = Variable(th.Tensor(np.random.uniform(0,1,(batch_size, noise_dim))))
        generator_out = self.layer(noise)
        # Keep the functional features
        adversarial_attack = raw_attack.detach().clone()
        for i in range(batch_size):
            adversarial_attack[i][nonfunctional_features_index] = generator_out[i]
        return adversarial_attack
