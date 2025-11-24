"""
Implements the standard SAE training scheme.
Integrated from korean-sparse-llm-features-open
"""
import torch
import torch.nn as nn

from collections import namedtuple
from abc import (
    ABC,
    abstractmethod,
)


__all__ = [
    'AutoEncoder',
    'StandardTrainer',
]


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """
    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(f_ghost)  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    @classmethod
    def from_pretrained(cls, path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = AutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p  # project away the parallel component of the gradient
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                p /= p.norm(dim=0, keepdim=True)  # renormalize the constrained parameters


class StandardTrainer():
    """
    Standard SAE training scheme.
    """
    def __init__(self,
                 dict_class=AutoEncoder,
                 activation_dim=512,
                 dict_size=64 * 512,
                 lr=1e-3,
                 l1_penalty=1e-1,
                 warmup_steps=1000,  # lr warmup period at start of training and after each resample
                 device='cpu',
                 resample_steps=None,  # how often to resample neurons
                 ):
        super().__init__()

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps

        self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps

        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = torch.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)
        if resample_steps is None:
            def warmup_fn(step):
                return min(step / warmup_steps, 1.)
        else:
            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)

    def resample_neurons(self, deads, activations):
        # activations: [batch_size, activation_dim]
        # self.ae.encoder.weight: [activation_dim, dict_size]
        # self.ae.decoder.weight: [dict_size, activation_dim]
        with torch.no_grad():
            if deads.sum() == 0:
                return

            print(f"resampling {deads.sum().item()} neurons")

            # 각 뉴런의 평균 activation 크기 계산
            _, features = self.ae(activations, output_features=True)  # [batch_size, dict_size]
            mean_activation = features.mean(dim=0)  # [dict_size]

            # activation이 가장 작은 뉴런들을 dead로 표시
            threshold = torch.quantile(mean_activation, 0.1)  # 하위 10% 뉴런 선택
            deads = mean_activation < threshold  # [dict_size] (boolean tensor)

            if deads.sum() == 0:
                return

            # 높은 재구성 오차를 가진 입력 샘플 선택
            losses = (activations - self.ae(activations)).norm(dim=-1)  # [batch_size]
            n_resample = min(deads.sum(), losses.shape[0])  # batch_size
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)  # [n_resample]
            sampled_vecs = activations[indices]  # [n_resample, activation_dim]

            # 살아있는 뉴런들의 평균 norm으로 스케일링
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()  # [n_alive, activation_dim]

            # dead 뉴런 재초기화
            sampled_vecs_normalized = sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2  # [n_dead, activation_dim]
            self.ae.decoder.weight[:, deads] = sampled_vecs_normalized.T  # [activation_dim, n_dead]
            self.ae.encoder.bias[deads] = 0.

            # Adam 옵티마이저 상태 초기화
            state_dict = self.optimizer.state_dict()['state']

            # encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.

            # encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.

            # decoder weight
            state_dict[3]['exp_avg'][:, deads] = 0.
            state_dict[3]['exp_avg_sq'][:, deads] = 0.

    # step is for interface compatibility
    def loss(self, x, step=None, logging=False, **kwargs):
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        # update steps_since_active
        if self.steps_since_active is not None:
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = l2_loss + self.l1_penalty * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple(
                'LossLog',
                ['x', 'x_hat', 'f', 'losses']
            )(x, x_hat, f, {
                'loss': loss.item(),
                'l2_loss': l2_loss.item(),
                'mse_loss': (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                'sparsity_loss': l1_loss.item(),
            })

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()

        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class': 'StandardTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'l1_penalty': self.l1_penalty,
            'warmup_steps': self.warmup_steps,
            'resample_steps': self.resample_steps,
            'device': self.device,
        }
