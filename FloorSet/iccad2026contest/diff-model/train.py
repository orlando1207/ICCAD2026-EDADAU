import os
import time
import random
import warnings
import argparse
import logging

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from models.model import GraphPlacer
from models.diffusion import MacroDiff
from utils.normalize import normalize, unnormalize, log_normalize
from utils.score import (
    get_overlap_tensor, 
    get_hpwl_tensor, 
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
parser.add_argument('-c', '--checkpoint_path', type=str, default='./checkpoint', 
                    help="Path to the model checkpoint (default: ./checkpoint)")
parser.add_argument('--data_path', type=str, default='./dataset', 
                    help="Path to the dataset (default: ./dataset)")
parser.add_argument('-r', '--result_path', type=str, default='./test', 
                    help="Path to the result (default: ./test)")
parser.add_argument('-b', '--batch_size', type=int, default=4, 
                    help="size of batch (default: 1)")
parser.add_argument('-t', '--timesteps', type=int, default=300, 
                    help="timesteps (default: 300)")
parser.add_argument('--start_epoch', type=int, default=0, 
                    help="start epoch (default: 0)")
parser.add_argument('--epochs', type=int, default=1000000, 
                    help="train epochs (default: 100000)")
parser.add_argument('--lr', type=float, default=1e-2, 
                    help="learning rate (default: 1e-2)")
parser.add_argument('--dropout', type=float, default=0., 
                    help="dropout rate (default: 0.)")
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')


class CircuitDataset(Dataset):
    def __init__(self, dataset_path, train=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.train = train
        self.data_list = []

        if train == True:
            design_file =  dataset_path + '/test.pt'
        else:
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


# Change sequences with different lengths to torch after padding 
def pad_and_stack(sequences):
    max_len = max(len(seq) for seq in sequences)
    padded = [torch.nn.functional.pad(seq, (0, 0, 0, max_len - len(seq))) for seq in sequences]
    return torch.stack(padded)


# Create chunks of a single sequence by sample unit
def process_chunks(data, split_sizes, num_io, batch_size):
    chunks = []
    start = 0
    for i in range(batch_size):
        end = split_sizes[i]
        chunk = data[start:end]
        chunks.append((chunk[:num_io[i]], chunk[num_io[i]:]))
        start = end
    return chunks


def main():
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger.info("Initializing model...")
    in_node_channels = 2
    in_net_channels = 2
    hidden_channels = 128
    edge_dim = 2

    model = GraphPlacer(in_node_channels, in_net_channels, hidden_channels, edge_dim)
    diffusion = MacroDiff(model, timesteps=args.timesteps)

    model.to(args.device)
    diffusion.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')

    if args.pretrained:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    logger.info("Loading datasets...")
    train_dataset = CircuitDataset(args.data_path, train=True)
    test_dataset = CircuitDataset(args.data_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    hpwl_dict = {}
    overlap_dict = {}

    ##### Get HPWL and overlap for baseline design #####
    for i, data in enumerate(test_loader):
        data = data.to(args.device)
        size = data['node']['size'].clone()
        pos = data['node']['pos'].clone()
        num_io = data['num_io']
        num_macro = data['num_macro']
        size_test = unnormalize(size, data['max_size'])
        pos_test = unnormalize(pos, data['max_size'])

        hpwl, _ = get_hpwl_tensor(data, pos_test)
        overlap, _ = get_overlap_tensor(data, size_test, pos_test) 
        hpwl_dict[design_list[i]] = hpwl
        overlap_dict[design_list[i]] = overlap

    logger.info("Start training...")
    for e in range(args.start_epoch, args.epochs):
        diffusion.model.train()
        total_loss = 0.0
        start_time = time.time()

        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
            loss = diffusion.compute_loss(data)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        logger.info(f"Epoch {e+1}/{args.epochs} | Loss: {total_loss:.4f} | Time: {end_time - start_time:.2f}s")

        if (e + 1) % 10 == 0:
            logger.info(f"Validation...")
            model.eval()
            total_sample_loss = 0.0
            hpwl_dict = {}
            overlap_dict = {}
            for i, data in enumerate(test_loader):
                data = data.to(args.device)
                num_io = data['num_io']
                num_macro = data['num_macro']
                size = data['node']['size'].clone()
                pos = data['node']['pos'].clone()
                size_test = unnormalize(size, data['max_size'])
                pos_test = unnormalize(pos, data['max_size'])

                io, x_0 = pos.split([num_io, num_macro])
                sample, _ = diffusion.sample_ddpm(data)

                d_target = sample.clone().detach().squeeze()
                x_init = torch.randn_like(x_0, requires_grad=True)
                x_optimizer = torch.optim.Adam([x_init], lr=0.01)
                for step in range(500):
                    pos_init = torch.cat([io, x_init], dim=0)
                    pos_init_test = unnormalize(pos_init, data['max_size'])

                    _, d_pred = get_hpwl_diff_test(data, pos_init_test)
                    d_pred = normalize(d_pred, data['max_size'][0]+data['max_size'][1])
                    d_pred = d_pred.squeeze()
                    d_loss = F.mse_loss(d_pred, d_target)

                    _, o_pred = get_overlap_diff(size_test[num_io:], pos_init_test[num_io:])
                    o_pred = o_pred / (size_test[num_io:,0]*size_test[num_io:,1])
                    o_cost = o_pred.mean()

                    beta = 1.0
                    loss = d_loss + beta * o_cost

                    x_optimizer.zero_grad()
                    loss.backward()
                    x_optimizer.step()
                    x_init.data.clamp_(-1, 1)

                x_0 = torch.cat([io, x_init], dim=0)
                pos_final = unnormalize(x_0, data['max_size'])
                pos_final = pos_final.clamp(min=torch.zeros_like(pos_final), max=data['max_size']-size_test)
                plot_design(pos_final, size_test, data['max_size'], save_dir=args.result_path, design=design_list[i], time=(e+1))

                hpwl, _ = get_hpwl_tensor(data, pos_final)
                overlap, _ = get_overlap_tensor(data, size_test, pos_final)
                hpwl_dict[design_list[i]] = hpwl
                overlap_dict[design_list[i]] = overlap
                
            logger.info(f"Saving model to {args.checkpoint_path}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{args.checkpoint_path}/epoch_{e+1}.ckpt')
        

if __name__ == '__main__':
    main()
    
