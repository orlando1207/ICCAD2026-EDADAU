import torch
import torch_scatter


def get_hpwl_tensor(data, pos_tensor):
    cell_pos_per_edge = pos_tensor[data['node', 'out', 'net'].edge_index[0]]
    offset_list = data['node', 'out', 'net'].edge_attr
    cell_pos_with_offset = cell_pos_per_edge + offset_list
    net_node_ids = data['node', 'out', 'net'].edge_index[1]

    x_min = torch_scatter.scatter(cell_pos_with_offset[:,0], net_node_ids, reduce='min')
    x_max = torch_scatter.scatter(cell_pos_with_offset[:,0], net_node_ids, reduce='max')
    y_min = torch_scatter.scatter(cell_pos_with_offset[:,1], net_node_ids, reduce='min')
    y_max = torch_scatter.scatter(cell_pos_with_offset[:,1], net_node_ids, reduce='max')

    width = x_max - x_min
    height = y_max - y_min
    hpwl_list = torch.stack((width, height), dim=1)
    hpwl = hpwl_list.sum()

    return hpwl, hpwl_list


def get_overlap_tensor(data, size_tensor, pos_tensor):
    cell_mins = pos_tensor
    cell_maxs = pos_tensor + size_tensor

    left = cell_mins[:, 0]
    right = cell_maxs[:, 0]
    bottom = cell_mins[:, 1]
    top = cell_maxs[:, 1]

    left_overlap = torch.maximum(left[:, None], left)
    right_overlap = torch.minimum(right[:, None], right)
    bottom_overlap = torch.maximum(bottom[:, None], bottom)
    top_overlap = torch.minimum(top[:, None], top)

    overlap_width = torch.clamp(right_overlap - left_overlap, min=0)
    overlap_height = torch.clamp(top_overlap - bottom_overlap, min=0)
    overlap_area = overlap_width * overlap_height
    self_area = size_tensor[:, 0] * size_tensor[:, 1]

    overlap_cost = (torch.sum(overlap_area) - torch.sum(self_area))
    overlap_list = (torch.sum(overlap_area, dim=1) - self_area)

    return overlap_cost, overlap_list


def get_hpwl_diff_test(data, pos_tensor, gamma=0.01, eps=1e-12):
    edge_index = data['node', 'out', 'net'].edge_index
    offset_list = data['node', 'out', 'net']['offset']
    net_node_ids = edge_index[1] 
    cell_indices = edge_index[0] 
    cell_pos = pos_tensor[cell_indices]
    cell_pos_with_offset = cell_pos + offset_list

    x = cell_pos_with_offset[:,0]
    y = cell_pos_with_offset[:,1]

    max_x = torch_scatter.scatter(x, net_node_ids, reduce='max')
    min_x = torch_scatter.scatter(x, net_node_ids, reduce='min')
    x_diff_max = x - max_x[net_node_ids]
    x_diff_min = x - min_x[net_node_ids]
    exp_max_x = torch.exp(x_diff_max / gamma)
    exp_min_x = torch.exp(-x_diff_min / gamma)
    sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_node_ids, reduce='sum')
    sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_node_ids, reduce='sum')
    sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_node_ids, reduce='sum')
    sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_node_ids, reduce='sum')
    wa_x_plus = max_x + sum_x_exp_max_x / (sum_exp_max_x + eps)
    wa_x_minus = min_x + sum_x_exp_min_x / (sum_exp_min_x + eps)
    wa_x = wa_x_plus - wa_x_minus

    max_y = torch_scatter.scatter(y, net_node_ids, reduce='max')
    min_y = torch_scatter.scatter(y, net_node_ids, reduce='min')
    y_diff_max = y - max_y[net_node_ids]
    y_diff_min = y - min_y[net_node_ids]
    exp_max_y = torch.exp(y_diff_max / gamma)
    exp_min_y = torch.exp(-y_diff_min / gamma)
    sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_node_ids, reduce='sum')
    sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_node_ids, reduce='sum')
    sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_node_ids, reduce='sum')
    sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_node_ids, reduce='sum')
    wa_y_plus = max_y + sum_y_exp_max_y / (sum_exp_max_y + eps)
    wa_y_minus = min_y + sum_y_exp_min_y / (sum_exp_min_y + eps)
    wa_y = wa_y_plus - wa_y_minus

    hpwl_list = wa_x + wa_y
    hpwl_list = hpwl_list.unsqueeze(-1)
    hpwl_approx = hpwl_list.sum()

    return hpwl_approx, hpwl_list


def get_hpwl_diff(data, pos_tensor, gamma=0.01, eps=1e-12):
    edge_index = data['node', 'out', 'net'].edge_index
    offset_list = data['node', 'out', 'net']['offset']
    split_sizes = torch.cumsum(data['net_size'], dim=0)
    length = data['leng']
    start = 0 
    hpwl_list_list = []
    hpwl_approx_list = []

    for i in range(len(pos_tensor)):
        end = split_sizes[i].item()
        net_node_ids = edge_index[1][start:end]
        net_node_ids = net_node_ids-net_node_ids.min()
        cell_indices = edge_index[0][start:end]
        cell_indices = cell_indices-cell_indices.min()
        cell_pos = pos_tensor[i][:length[i]]
        cell_pos = cell_pos[cell_indices]
        offset = offset_list[start:end]
        cell_pos_with_offset = cell_pos + offset
        
        x = cell_pos_with_offset[:,0]
        y = cell_pos_with_offset[:,1]
        max_x = torch_scatter.scatter(x, net_node_ids, reduce='max')
        min_x = torch_scatter.scatter(x, net_node_ids, reduce='min')
        x_diff_max = x - max_x[net_node_ids]
        x_diff_min = x - min_x[net_node_ids]
        exp_max_x = torch.exp(x_diff_max / gamma)
        exp_min_x = torch.exp(-x_diff_min / gamma)
        sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_node_ids, reduce='sum')
        sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_node_ids, reduce='sum')
        sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_node_ids, reduce='sum')
        sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_node_ids, reduce='sum')
        wa_x_plus = max_x + sum_x_exp_max_x / (sum_exp_max_x + eps)
        wa_x_minus = min_x + sum_x_exp_min_x / (sum_exp_min_x + eps)
        wa_x = wa_x_plus - wa_x_minus

        max_y = torch_scatter.scatter(y, net_node_ids, reduce='max')
        min_y = torch_scatter.scatter(y, net_node_ids, reduce='min')
        y_diff_max = y - max_y[net_node_ids]
        y_diff_min = y - min_y[net_node_ids]
        exp_max_y = torch.exp(y_diff_max / gamma)
        exp_min_y = torch.exp(-y_diff_min / gamma)
        sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_node_ids, reduce='sum')
        sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_node_ids, reduce='sum')
        sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_node_ids, reduce='sum')
        sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_node_ids, reduce='sum')
        wa_y_plus = max_y + sum_y_exp_max_y / (sum_exp_max_y + eps)
        wa_y_minus = min_y + sum_y_exp_min_y / (sum_exp_min_y + eps)
        wa_y = wa_y_plus - wa_y_minus

        hpwl_list = wa_x + wa_y
        hpwl_list = hpwl_list.unsqueeze(-1)
        hpwl_approx = hpwl_list.sum()

        hpwl_list_list.append(hpwl_list)
        hpwl_approx_list.append(hpwl_approx) 
        start = end

    return hpwl_approx_list, hpwl_list_list


def get_overlap_diff(size_tensor, pos_tensor, gamma=0.01):
    def smooth_max(a, b):
        max_val = torch.max(a, b)
        return gamma * (torch.log(torch.exp((a - max_val) / gamma) + torch.exp((b - max_val) / gamma))) + max_val

    def smooth_min(a, b):
        min_val = torch.min(a, b)
        return -gamma * (torch.log(torch.exp((-a + min_val) / gamma) + torch.exp((-b + min_val) / gamma))) + min_val
    
    cell_mins = pos_tensor
    cell_maxs = pos_tensor + size_tensor

    left    = cell_mins[:, 0]
    right   = cell_maxs[:, 0]
    bottom  = cell_mins[:, 1]
    top     = cell_maxs[:, 1]

    left_overlap    = smooth_max(left[:, None], left)
    right_overlap   = smooth_min(right[:, None], right)
    bottom_overlap  = smooth_max(bottom[:, None], bottom)
    top_overlap     = smooth_min(top[:, None], top)

    overlap_width   = torch.clamp(right_overlap - left_overlap, min=0)
    overlap_height  = torch.clamp(top_overlap - bottom_overlap, min=0)
    overlap_area    = overlap_width * overlap_height

    self_mask       = torch.eye(overlap_area.size(0), dtype=torch.bool, device=overlap_area.device)
    overlap_area    = overlap_area.masked_fill(self_mask, 0.)

    overlap_cost    = torch.sum(overlap_area)
    overlap_list    = torch.sum(overlap_area, dim=1)

    return overlap_cost, overlap_list