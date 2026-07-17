import os
import time
import random
import warnings
import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from models.model import GraphPlacer
from models.diffusion import MacroDiff
from utils.normalize import normalize, unnormalize, log_normalize
from utils.score import (
    get_overlap_diff, 
    get_hpwl_diff_test,
)
from utils.plot import plot_design


design_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4', 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', ]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int, default=0, 
                    help="CUDA device index (default: 0)")
parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/checkpoint.ckpt', 
                    help="Path to the model checkpoint (default: ./checkpoint/checkpoint.ckpt)")
parser.add_argument('--data_path', type=str, default='./dataset', 
                    help="Path to the dataset (default: ./dataset)")
parser.add_argument('-r', '--result_path', type=str, default='./result', 
                    help="Path to the result (default: ./result)")
parser.add_argument('--timesteps', type=int, default=300, 
                    help="timesteps (default: 300)")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')


class CircuitDataset(Dataset):
    def __init__(self, dataset_path, train=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.data_list = []

        design_file =  dataset_path + '/test.pt'
        
        if not os.path.exists(design_file):
            raise FileNotFoundError(f'Data file {design_file} does not exist.')
        data = torch.load(design_file)
        for design in design_list:
            for sample in data[design]:
                sample['node']['pos'] = normalize(sample['node']['pos'], sample['max_size'])
                sample['node']['size'] = normalize(sample['node']['size'], sample['max_size'])
                sample['leng'] = len(sample['node']['pos'])
                sample['node']['degree_size'] = len(sample['net']['degree'])
                sample['net']['degree'] = log_normalize(sample['net']['degree'])
                sample['net_size'] = len(sample['node', 'out', 'net'].edge_attr)
                sample['node', 'out', 'net']['offset'] = sample['node', 'out', 'net'].edge_attr.clone()
                sample['node', 'out', 'net'].edge_attr = normalize(sample['node', 'out', 'net'].edge_attr, sample['max_size'])
                sample['net', 'in', 'node']['offset'] = sample['net', 'in', 'node'].edge_attr.clone()
                sample['net', 'in', 'node'].edge_attr = normalize(sample['net', 'in', 'node'].edge_attr, sample['max_size'])
                sample['node'].x = sample['node']['pos'].clone()
                sample['net'].x = sample['net']['degree'].clone()
                self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def main():
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger.info("Initializing model...")
    in_node_channels = 2
    in_net_channels = 2
    hidden_channels = 128
    edge_dim = 2

    model = GraphPlacer(in_node_channels, in_net_channels, hidden_channels, edge_dim)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    diffusion = MacroDiff(model, timesteps=args.timesteps)
    model.to(args.device)
    diffusion.to(args.device)

    test_dataset = CircuitDataset(args.data_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        
    model.eval()
    for i, data in enumerate(test_loader):
        design = design_list[i]

        data = data.to(args.device)
        num_io = data['num_io']
        num_macro = data['num_macro']
        size = data['node']['size'].clone()
        pos = data['node']['pos'].clone()
        degree = data['net']['degree'].clone()
        size_test = unnormalize(size, data['max_size'])
        pos_test = unnormalize(pos, data['max_size'])

        _, d_0 = get_hpwl_diff_test(data, pos_test)
        mask = (d_0 != 0.).squeeze()

        io, x_0 = pos.split([num_io, num_macro])
        total_overlap = 0.0
        for j in range(0, 10):
            sample, x_t = diffusion.sample_ddpm(data, seed=j)
            sample = torch.where(mask.unsqueeze(1), sample, -1.0*torch.ones_like(sample))
            d_target = sample.clone().detach().squeeze()
            x_init = ((torch.rand_like(x_0) - 0.5) * 1.0).detach().requires_grad_()

            x_optimizer = torch.optim.Adam([x_init], lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(x_optimizer, lr_lambda=lambda step: 1 - step / 500)

            for step in range(500):
                x_tanh = F.tanh(x_init)
                pos_init = torch.cat([io, x_tanh], dim=0)
                pos_init_test = unnormalize(pos_init, data['max_size'])
                
                d_cost, d_pred = get_hpwl_diff_test(data, pos_init_test)
                d_pred = normalize(d_pred, data['max_size'][0]+data['max_size'][1])
                d_cost = d_cost / (data['max_size'][0]+data['max_size'][1])
                d_pred = d_pred.squeeze()
                d_loss = F.mse_loss(d_pred[mask], d_target[mask])

                o_cost, o_pred = get_overlap_diff(size_test[num_io:], pos_init_test[num_io:])
                o_cost = o_cost / (data['max_size'][0]*data['max_size'][1])


                if design == 'adaptec2' or design == 'bigblue4':
                    alpha = 0.0005
                else:
                    alpha = 0.001
                beta = 0.1 * step / 500
                loss = d_loss + alpha * d_cost + beta * o_cost

                x_optimizer.zero_grad()
                loss.backward()
                x_optimizer.step()
                scheduler.step()
                x_init.data.clamp_(-1, 1)

            total_overlap += o_cost.item()

            pos_0 = torch.cat([io, x_init], dim=0)
            pos_0_test = unnormalize(pos_0, data['max_size'])
            pos_0_test = pos_0_test.clamp(min=torch.zeros_like(pos_0_test), max=data['max_size']-size_test)
            plot_design(pos_0_test, size_test, data['max_size'], save_dir=args.result_path, design=design_list[i], time=j)

if __name__ == '__main__':
    main()