# =================================================================
#
#                       WGAN-GP Class
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
import numpy as np
import torch as th
import pandas as pd
import torch.nn as nn

from wgan.discriminator import Discriminator
from wgan.generator import Generator
from wgan.utils import create_batch, compute_gradient_penalty
from torch.autograd import Variable


# ==================> Constants

FUNCTIONAL_FEATURES = [
 ' min_seg_size_forward',' Bwd Header Length',' Destination Port'
 'Init_Win_bytes_forward',' Init_Win_bytes_backward',' Bwd Packets/s'
 'Total Length of Fwd Packets',' Subflow Fwd Bytes',' Max Packet Length'
 'Bwd Packet Length Max',' Avg Bwd Segment Size',' Bwd Packet Length Mean'
 ' Fwd Packet Length Max',' Average Packet Size',' Packet Length Std'
 ' Packet Length Mean',' Bwd Packet Length Std',' Bwd Packet Length Min'
 ' Fwd Packet Length Std',' Fwd Packet Length Min',' Min Packet Length'
 ' Fwd Packet Length Mean',' Avg Fwd Segment Size',' act_data_pkt_fwd'
 ' Total Fwd Packets','Subflow Fwd Packets',' Total Backward Packets']


# ==================> Class

class WGAN_GP:
    """
    Wasserstein GAN with Gradient Penalty.
    """

    def __init__(self,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 ids_model: any
                 ):
        """
        Initialize the WGAN-GP class.

        :param x_train: pd.DataFrame, training data.
        :param y_train: pd.DataFrame, training labels.
        :param x_test: pd.DataFrame, test data.
        :param y_test: pd.DataFrame, test labels.
        :param ids_model: IDSModel, (RF, SVM, etc).
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.ids_model = ids_model
        self.NON_FUNCTIONAL_FEATURES_IDEXES = [self.x_train.columns.get_loc(c) for c in self.x_train.columns if c not in FUNCTIONAL_FEATURES][:-1]

        # Separe attack and normal data
        self.normal_data = x_train[y_train == 0]
        self.attacks_data = x_train[y_train == 1]

        # WGAN-GP parameters
        self.batch_size = 256
        self.critic_iters = 5  # The number of iterations of the critic per generator iteration
        self.lambda_ = 10  # Gradient penalty lambda hyperparameter
        self.max_epochs = 100
        self.learning_rate = 0.0001
        self.clamp = 0.01  # Lower and upper clip value for disc. weights
        self.discriminator_input_dim = len(self.x_train.columns)
        self.discriminator_output_dim = 1
        self.generator_input_dim = len(self.x_train.columns)
        self.generator_output_dim = len(self.x_train.columns)

    def _init_generator_and_discriminator(self):
        """
        Initialize the generator and discriminator.
        """
        self.generator = Generator(self.generator_input_dim, self.generator_output_dim)
        self.discriminator = Discriminator(self.discriminator_input_dim, self.discriminator_output_dim)

    def _init_optimizer(self):
        """
        Initialize the optimizer.
        """
        self.optimizer_G = th.optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_D = th.optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)
        self.generator.train()
        self.discriminator.train()
        self.ids_model.eval()

    def _init_device(self):
        """
        Initialize the device (CPU or GPU).
        """
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    
    def train_generator(self, generator, discriminator, optimizer_G, noise_dim, attack_traffic):
        """
        Train the generator.
        """
        for p in discriminator.parameters():  
            p.requires_grad = False
        optimizer_G.zero_grad()
        # GAN-G Generate Adversarial Attack
        adversarial_attack = generator(noise_dim, attack_traffic, self.NON_FUNCTIONAL_FEATURES_IDEXES)
        # GAN-D predict, GAN-G update parameter
        D_pred = discriminator(adversarial_attack)
        g_loss = -th.mean(D_pred)
        g_loss.backward()
        optimizer_G.step()
        return g_loss

    def train_discriminator(self, discriminator, ids_model, generator, critic_iters, clamp, optimizer_D, normal_b, noise_dim, attack_traffic):
        run_d_loss = 0
        cnt = 0
        for p in discriminator.parameters(): 
            p.requires_grad = True
        for c in range(critic_iters):
            optimizer_D.zero_grad()
            for p in discriminator.parameters():
                p.data.clamp_(-clamp, clamp)
            # GAN-G Generate Adversarial Attack
            adversarial_attack = generator(noise_dim, attack_traffic, self.NON_FUNCTIONAL_FEATURES_IDEXES)
            ids_input = th.cat((adversarial_attack,normal_b))
            l = list(range(len(ids_input)))
            np.random.shuffle(l)
            ids_input = ids_input[l]
            # IDS predict
            ids_pred = ids_model.predict(ids_input.detach().numpy())
            pred_attack = ids_input[ids_pred == 1]
            pred_normal = ids_input[ids_pred == 0]
            if len(pred_attack) == 0:
                cnt += 1
                break
            # Make GAN-D input
            D_noraml = discriminator(Variable(th.Tensor(pred_normal)))
            D_attack= discriminator(Variable(th.Tensor(pred_attack)))
            # Loss and Update Parameter
            loss_normal = th.mean(D_noraml)
            loss_attack = th.mean(D_attack)
            gradient_penalty = compute_gradient_penalty(discriminator, normal_b.data, adversarial_attack.data)
            d_loss = loss_attack - loss_normal + self.lambda_ * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            run_d_loss += d_loss.item()
        return run_d_loss, cnt

    
    def cal_dr(self, ids_model, normal, raw_attack, adversarial_attack):
        # Make data to feed IDS contain: Attack & Normal
        o_ids_input = th.cat((raw_attack, normal))
        a_ids_input = th.cat((adversarial_attack,normal))
        # Shuffle Input
        l = list(range(len(a_ids_input)))
        np.random.shuffle(l)
        o_ids_input = o_ids_input[l]
        a_ids_input = a_ids_input[l]
        # IDS Predict Label
        o_pred_label = th.Tensor(ids_model.predict(o_ids_input))
        a_pred_label = th.Tensor(ids_model.predict(a_ids_input))
        # True Label
        ids_true_label = np.r_[np.ones(self.batch_size),np.zeros(self.batch_size)][l]
        # Calc DR
        tn1, fn1, fp1, tp1 = confusion_matrix(ids_true_label,o_pred_label).ravel()
        tn2, fn2, fp2, tp2 = confusion_matrix(ids_true_label,a_pred_label).ravel()
        origin_dr = tp1/(tp1 + fp1)
        adversarial_dr = tp2/(tp2 + fp2)
        return origin_dr, adversarial_dr


    def train(self) -> (list[float], list[float]):
        """
        Train the WGAN-GP model.

        :return: Generator and discriminator losses.
        """
        self._init_generator_and_discriminator()
        self._init_optimizer()
        self._init_device()

        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Save losses
        d_losses,g_losses = [],[]
        o_dr, a_dr = [],[]

        batch_attack = create_batch(attacks_data, BATCH_SIZE)
        cnt = -5

        # Train
        for epoch in range(self.max_epochs):
            # Create normal batch
            batch_normal = create_batch(self.normal_data.values, self.batch_size)

            # Generator and discriminator losses
            run_g_loss = 0.
            run_d_loss = 0.

            cnt = 0
            run_g_loss = 0.
            run_d_loss = 0.
            epoch_o_drs, epoch_a_drs = [], []

            for i, bn in enumerate(batch_normal):
                normal_b = th.Tensor(bn.astype("float64")).to(self.device)
                attack_traffic  = Variable(th.Tensor(batch_attack[i % len(batch_attack)]))
                # Train Generator
                g_loss = self.train_generator(self.generator, self.discriminator, self.optimizer_G, self.generator_input_dim, attack_traffic)
                run_g_loss += g_loss.item()

                # Train Discriminator
                d_loss, current_cnt = self.train_discriminator(self.discriminator, self.ids_model, self.generator, self.critic_iters, self.clamp, self.optimizer_D,
                    normal_b, self.generator_input_dim, attack_traffic)

                run_d_loss += d_loss
                cnt += current_cnt

                 # CALC Epoch DR
                adversarial_attack = self.generator(self.generator_input_dim, attack_traffic, self.NON_FUNCTIONAL_FEATURES_IDEXES).detach()
                origin_dr, adversarial_dr = self.cal_dr(self.ids_model, normal_b, attack_traffic, adversarial_attack)
                epoch_o_drs.append(origin_dr)
                epoch_a_drs.append(adversarial_dr)

            if cnt >= (len(normal_data)/BATCH_SIZE):
                print("Not exist predicted attack traffic")
                break
            d_losses.append(run_d_loss/CRITIC_ITERS)
            g_losses.append(run_g_loss)
            epoch_o_dr = np.mean(epoch_o_drs)
            epoch_a_dr = np.mean(epoch_a_drs)
            o_dr.append(epoch_o_dr)
            a_dr.append(epoch_a_dr)

            print("Epoch: %d, G_loss: %.4f, D_loss: %.4f, Origin DR: %.4f, Adversarial DR: %.4f" % (epoch, run_g_loss, run_d_loss/CRITIC_ITERS, epoch_o_dr, epoch_a_dr))
        return g_losses, d_losses


    def save_models(self, path):
        """
        Save the generator and discriminator models.

        :param path: Path to save the models.
        """
        th.save(self.generator.state_dict(), path + "generator.pth")
        th.save(self.discriminator.state_dict(), path + "discriminator.pth")

    def load_models(self, path) -> (Generator, Discriminator):
        """
        Load the generator and discriminator models.

        :param path: Path to load the models.
        """
        self._init_generator_and_discriminator()
        self._init_device()
        self.generator.load_state_dict(th.load(path + "generator.pth"))
        self.discriminator.load_state_dict(th.load(path + "discriminator.pth"))
        return self.generator, self.discriminator
