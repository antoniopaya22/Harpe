# =================================================================
#
#                       WGAN-GP Discriminator
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
import torch as th

from torch import nn


# ==================> Class
class Discriminator(nn.Module):
    """
    Discriminator class for WGAN-GP.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Constructor.

        :param input_dim: int, dimension of the input space.
        :param output_dim: int, dimension of the output space.
        """
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.layer(x)