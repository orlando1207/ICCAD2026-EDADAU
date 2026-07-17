import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.normalize import normalize, unnormalize
from utils.score import get_hpwl_diff, get_hpwl_diff_test


def extract(a, t, x_shape):
    b = t.shape[0] if len(t.shape) > 0 else 1
    out = a.gather(-1, t.long().view(-1))
    return out.view(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, start, end):
    return torch.linspace(start, end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start, end, tau = 1.):
    t = torch.linspace(0, 1, timesteps+1, dtype=torch.float64)
    v_start = torch.sigmoid(torch.tensor(start / tau))
    v_end = torch.sigmoid(torch.tensor(end / tau))
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MacroDiff(nn.Module):
    def __init__(
        self,
        model,
        timesteps=300,
        beta_schedule="cosine",
        beta_start=-3.,
        beta_end=3.,
    ):
        super(MacroDiff, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.)

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer('alphas_bar', alphas_bar)
        register_buffer('alphas_bar_prev', alphas_bar_prev)
        register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        register_buffer('log_one_minus_alphas_bar', torch.log(1. - alphas_bar))
        register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        register_buffer('sqrt_recip_alphas_bar_m1', torch.sqrt((1. / alphas_bar) - 1.))

        posterior_var = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        register_buffer('posterior_var', posterior_var)
        register_buffer('posterior_logvar_clipped', torch.log(torch.cat([posterior_var[[1]], posterior_var[1:]]).clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * alphas_bar_prev / (1. - alphas_bar))
        register_buffer('posterior_mean_coef2', (1. - alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar))

    def to(self, device):
        super(MacroDiff, self).to(device)
        for name, buffer in self._buffers.items():
            if buffer is not None:
                self.register_buffer(name, buffer.to(device))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - 
            extract(self.sqrt_recip_alphas_bar_m1, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_0, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + 
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = extract(self.posterior_var, t, x_t.shape)
        posterior_logvar = extract(self.posterior_logvar_clipped, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_logvar
    
    def q_sample(self, x_0, t, noise):
        return (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
    
    def compute_loss(self, data):
        num_io = data['num_io']
        num_macro = data['num_macro']
        batch_size = len(num_io)

        size = data['node']['size'].clone()
        pos = data['node']['pos'].clone()
        degree = data['net']['degree'].clone()

        def pad_and_stack(sequences):
            max_len = max(len(seq) for seq in sequences)
            padded = [torch.nn.functional.pad(seq, (0, 0, 0, max_len - len(seq)), value=-1.) for seq in sequences]
            return torch.stack(padded)

        def process_chunks(data, split_sizes, num_io, batch_size):
            chunks = []
            start = 0
            for i in range(batch_size):
                end = split_sizes[i]
                chunk = data[start:end]
                chunks.append((chunk[:num_io[i]], chunk[num_io[i]:]))
                start = end
            return chunks
        
        split_sizes = torch.cumsum(num_io + num_macro, dim=0)
        chunks = process_chunks(pos, split_sizes, num_io, batch_size)
        io_list, x_0_list = zip(*chunks)
        io = pad_and_stack(io_list)
        x_0 = pad_and_stack(x_0_list)
        pos = torch.cat([io, x_0], dim=1)

        chunks = process_chunks(size, split_sizes, num_io, batch_size)
        io_size_list, x_0_size_list = zip(*chunks)
        io_size = pad_and_stack(io_size_list)
        x_0_size = pad_and_stack(x_0_size_list)
        size = torch.cat([io_size, x_0_size], dim=1)

        def process_positions(x,data_max_size):
            return [unnormalize(x[i], data_max_size[2*i:2*i+2]).clone().detach() for i in range(len(x))]

        pos_0_test = process_positions(pos, data['max_size'])
        _, d_0_test = get_hpwl_diff(data, pos_0_test)

        def normalize_list(data_list, data_max_size):
            return [normalize(data, data_max_size[2*i:2*i+2][0] + data_max_size[2*i:2*i+2][1]) for i, data in enumerate(data_list)]
        
        d_0 = normalize_list(d_0_test, data['max_size'])
        d_0 = pad_and_stack(d_0)

        t = torch.randint(0, self.timesteps, (x_0.size(0),), device=x_0.device)
        noise = torch.randn_like(d_0)
        d_t = self.q_sample(d_0, t, noise)

        degree_split_sizes = torch.cumsum(data['node']['degree_size'], dim=0)
        degree_chunks = process_chunks(degree, degree_split_sizes, [0] * batch_size, batch_size)
        degree = pad_and_stack([chunk[1] for chunk in degree_chunks])

        data['node'].x = size
        data['net'].x = torch.cat([d_t, degree], dim=2)
    
        pred_net, pred_cell = self.model(data, t)
        net_loss_all = F.mse_loss(pred_net, noise, reduction='none')
        mask = (d_0 != -1.)
        net_loss = (net_loss_all * mask).sum() / mask.sum()

        return net_loss

    
    @torch.inference_mode()
    def sample_ddpm(self, data, seed=None):
        num_io = data['num_io']
        num_macro = data['num_macro']

        size = data['node']['size'].clone()
        pos = data['node']['pos'].clone()
        degree = data['net']['degree'].clone()

        io, x_0 = pos.split([num_io, num_macro])
        pos_0_test = unnormalize(pos, data['max_size']).clone().detach()
        size_test = unnormalize(size, data['max_size']).clone().detach()

        _, d_0_test = get_hpwl_diff_test(data, pos_0_test)
        d_0 = normalize(d_0_test, data['max_size'][0]+data['max_size'][1]).unsqueeze(1)

        t = torch.randint(0, self.timesteps, (1,), device=x_0.device)
        noise = torch.randn_like(d_0)
        if seed is not None:
            torch.manual_seed(seed)
            d_t = torch.randn_like(d_0)
        else:
            torch.manual_seed(0)
            d_t = self.q_sample(d_0, t, noise)

        d_t = d_t.squeeze(-1)
        x_list = [d_t.clone()]
        for t in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total = self.timesteps):
            t = torch.tensor([t], device=d_t.device)
            data['node'].x = size
            data['net'].x = torch.cat([d_t, degree], dim=1)
            
            pred_net, _ = self.model(data, t)
            pred_noise = pred_net.squeeze(0)
            pred_d0 = self.predict_start_from_noise(d_t, t, pred_noise)
            pred_d0 = pred_d0.squeeze(0).clamp(-1, 1)

            model_mean, model_var, model_logvar = self.q_posterior(pred_d0, d_t, t)
            noise = torch.randn_like(pred_d0) if t > 0 else 0.
            d_t = model_mean + (0.5 * model_logvar).exp() * noise
            x_list.append(d_t.clone())

        return d_t, x_list

