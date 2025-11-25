"""
Implements the training scheme for a gated SAE described in https://arxiv.org/abs/2404.16014
Integrated from korean-sparse-llm-features-open
"""
import torch
import torch.nn as nn

from collections import namedtuple


__all__ = [
    'GatedAutoEncoder',
    'GatedTrainer',
]


class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr, betas=(0., 0.999))
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


class GatedAutoEncoder(nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """
    def __init__(self, activation_dim, dict_size, initialization='default', device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(torch.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(torch.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(torch.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(torch.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == 'default':
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        nn.init.zeros_(self.decoder_bias)
        nn.init.zeros_(self.r_mag)
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, return_gate=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)
        f = f_gate * f_mag

        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f):
        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class GatedTrainer():
    """
    Gated SAE training scheme with p-annealing.
    Supports multi-GPU training via DataParallel.
    """
    def __init__(self,
                 dict_class=GatedAutoEncoder,
                 activation_dim=512,
                 dict_size=64 * 512,
                 lr=3e-4,
                 warmup_steps=1000,  # lr warmup period at start of training and after each resample
                 sparsity_function='Lp^p',  # Lp or Lp^p
                 initial_sparsity_penalty=1e-1,  # equal to l1 penalty in standard trainer
                 anneal_start=15000,  # step at which to start annealing p
                 anneal_end=None,  # step at which to stop annealing, defaults to steps-1
                 p_start=1,  # starting value of p (constant throughout warmup)
                 p_end=0,  # annealing p_start to p_end linearly after warmup_steps, exact endpoint excluded
                 n_sparsity_updates=10,  # number of times to update the sparsity penalty, at most steps-anneal_start times
                 sparsity_queue_length=10,  # number of recent sparsity loss terms, onle needed for adaptive_sparsity_penalty
                 resample_steps=None,  # number of steps after which to resample dead neurons
                 total_steps=None,  # total number of steps to train for
                 device='cuda',  # Primary device (for backwards compatibility)
                 devices=None,  # List of devices for multi-GPU training
                 ):

        # initialize dictionary
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.ae = dict_class(activation_dim, dict_size)

        # Handle device configuration
        # devices takes precedence over device
        if devices is not None and len(devices) > 0:
            self.devices = devices
            self.device = devices[0]  # Primary device
            self.multi_gpu = len(devices) > 1
        else:
            self.device = device
            self.devices = [device]
            self.multi_gpu = False

        self.ae.to(self.device)

        # Wrap with DataParallel if multiple GPUs
        if self.multi_gpu:
            # Extract device IDs from device strings (e.g., "cuda:2" -> 2)
            device_ids = [int(d.split(':')[1]) if ':' in d else 0 for d in self.devices]
            self.ae = nn.DataParallel(self.ae, device_ids=device_ids)
            print(f"Multi-GPU enabled: {self.devices} (device_ids: {device_ids})")

        self.lr = lr
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end if anneal_end is not None else total_steps
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start  # p is set in self.loss()
        self.next_p = None  # set in self.loss()
        self.lp_loss = None  # set in self.loss()
        self.scaled_lp_loss = None  # set in self.loss()
        if n_sparsity_updates == 'continuous':
            self.n_sparsity_updates = self.anneal_end - anneal_start + 1
        else:
            self.n_sparsity_updates = n_sparsity_updates
        self.sparsity_update_steps = torch.linspace(anneal_start, self.anneal_end, self.n_sparsity_updates, dtype=int)
        self.p_values = torch.linspace(p_start, p_end, self.n_sparsity_updates)
        self.p_step_count = 0
        self.sparsity_coeff = initial_sparsity_penalty  # alpha
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = []

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.logging_parameters = ['p', 'next_p', 'lp_loss', 'scaled_lp_loss', 'sparsity_coeff']

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = torch.zeros(self.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None

        # Get the underlying model (handle DataParallel wrapping)
        base_model = self.ae.module if self.multi_gpu else self.ae
        self.optimizer = ConstrainedAdam(self.ae.parameters(), base_model.decoder.parameters(), lr=lr)
        if resample_steps is None:
            def warmup_fn(step):
                return min(step / warmup_steps, 1.)
        else:
            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)

    @property
    def base_model(self):
        """Get the underlying model, unwrapping DataParallel if needed."""
        return self.ae.module if self.multi_gpu else self.ae

    def resample_neurons(self, deads, activations):
        # activations: [batch_size, activation_dim]
        # base_model.encoder.weight: [activation_dim, dict_size]
        # base_model.decoder.weight: [dict_size, activation_dim]
        base_model = self.base_model
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
            alive_norm = base_model.encoder.weight[~deads].norm(dim=-1).mean()  # [n_alive, activation_dim]

            # dead 뉴런 재초기화
            sampled_vecs_normalized = sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)
            base_model.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2  # [n_dead, activation_dim]
            base_model.decoder.weight[:, deads] = sampled_vecs_normalized.T  # [activation_dim, n_dead]
            base_model.encoder.bias[deads] = 0.

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

    def lp_norm(self, f, p):
        norm_sq = f.pow(p).sum(dim=-1)
        if self.sparsity_function == 'Lp^p':
            return norm_sq.mean()
        elif self.sparsity_function == 'Lp':
            return norm_sq.pow(1 / p).mean()
        else:
            raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")

    def loss(self, x, step, logging=False, **kwargs):
        base_model = self.base_model
        f, f_gate = base_model.encode(x, return_gate=True)
        x_hat = base_model.decode(f)
        x_hat_gate = f_gate @ base_model.decoder.weight.detach().T + base_model.decoder_bias.detach()

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        fs = f_gate  # feature activation that we use for sparsity term
        lp_loss = self.lp_norm(fs, self.p)
        scaled_lp_loss = lp_loss * self.sparsity_coeff
        self.lp_loss = lp_loss
        self.scaled_lp_loss = scaled_lp_loss

        if self.next_p is not None:
            lp_loss_next = self.lp_norm(fs, self.next_p)
            self.sparsity_queue.append([self.lp_loss.item(), lp_loss_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.sparsity_queue_length:]

        if step in self.sparsity_update_steps:

            # check to make sure we don't update on repeat step:
            if step >= self.sparsity_update_steps[self.p_step_count]:

                # adapt sparsity penalty alpha
                if self.next_p is not None:
                    local_sparsity_new = torch.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_old = torch.tensor([i[1] for i in self.sparsity_queue]).mean()
                    self.sparsity_coeff = self.sparsity_coeff * (local_sparsity_new / local_sparsity_old).item()

                # update p
                self.p = self.p_values[self.p_step_count].item()
                if self.p_step_count < self.n_sparsity_updates - 1:
                    self.next_p = self.p_values[self.p_step_count + 1].item()
                else:
                    self.next_p = self.p_end
                self.p_step_count += 1

        # update dead feature count
        if self.steps_since_active is not None:

            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = L_recon + scaled_lp_loss + L_aux

        if not logging:
            return loss
        else:
            return namedtuple(
                'LossLog',
                ['x', 'x_hat', 'f', 'losses']
            )(x, x_hat, f, {
                'loss': loss.item(),
                'mse_loss': L_recon.item(),
                'aux_loss': L_aux.item(),
                'p': self.p,
                'next_p': self.next_p,
                'lp_loss': lp_loss.item(),
                'sparsity_loss': scaled_lp_loss.item(),
                'sparsity_coeff': self.sparsity_coeff,
            })

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step, logging=False)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == self.resample_steps - 1:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'trainer_class': 'GatedAnnealTrainer',
            'dict_class': 'GatedAutoEncoder',
            'activation_dim': self.activation_dim,
            'dict_size': self.dict_size,
            'lr': self.lr,
            'sparsity_function': self.sparsity_function,
            'sparsity_penalty': self.sparsity_coeff,
            'p_start': self.p_start,
            'p_end': self.p_end,
            'anneal_start': self.anneal_start,
            'sparsity_queue_length': self.sparsity_queue_length,
            'n_sparsity_updates': self.n_sparsity_updates,
            'warmup_steps': self.warmup_steps,
            'resample_steps': self.resample_steps,
            'total_steps': self.total_steps,
            'device': self.device,
            'devices': self.devices,
            'multi_gpu': self.multi_gpu,
        }
