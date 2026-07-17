import matplotlib.pyplot as plt


def plot_design(pos, size, max_size, save_dir=None, design=None, time=None):
    pos_cpu = pos.cpu()
    size_cpu = size.cpu()
    max_size_cpu = max_size.cpu()

    fig, ax = plt.subplots()

    for i in range(pos.shape[0]):
        x = pos_cpu[i, 0].item()
        y = pos_cpu[i, 1].item()
        width = size_cpu[i, 0].item()
        height = size_cpu[i, 1].item()
        rect = plt.Rectangle((x, y), width, height, facecolor='blue', edgecolor='none', alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, max_size_cpu[0])
    ax.set_ylim(0, max_size_cpu[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    if save_dir is not None:
        filename = f'{save_dir}/{design}.png' if time is None else f'{save_dir}/{design}_{time}.png'
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()