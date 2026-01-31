# Assuming access to a dataset of media, we will generate data pairs of sparse CT measurements and "full" CT measurements
import argparse
import torch
import os
from solvers.torch_eit_fem_solver import Mesh, V_h
from solvers.torch_eit_fem_solver.utils import dtn_from_sigma
from scipy import io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate data pairs of conductivities and dtn maps.")
    parser.add_argument('--problem', type=str, default='circles', help='Dataset to use')
    parser.add_argument('--mesh', type=str, default='default', help='Mesh to use')

    args = parser.parse_args()

    img_size = 128
    dataset_source = torch.load(f"data/eit-{args.problem}-dataset-{img_size}.pt")

    dataset = {}

    # load mesh
    data_root = 'mesh-data'
    assert args.mesh in ['default', 'refined'], "mesh must be either 'default' or 'refined'"
    if args.mesh == 'default':
        mesh_file = 'mesh_128_h05.mat'
    elif args.mesh == 'refined':
        mesh_file = 'completed_new_mesh.mat' 

    original_size = 128
    pad_size = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    mat_fname  = os.path.join(data_root, mesh_file)
    mat_contents = sio.loadmat(mat_fname)

    p = torch.tensor(mat_contents['p'], dtype=torch.float64)
    t = torch.tensor(mat_contents['t']-1, dtype=torch.long)
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,))-1, dtype=torch.long)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,))-1, dtype=torch.long)

    p = p.to(device)
    t = t.to(device)
    vol_idx = vol_idx.to(device)
    bdy_idx = bdy_idx.to(device)

    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)

    for split in ['train', 'val', 'test']:
        images = dataset_source[split]  # shape (N, 1, H, W)
        N, C, H, W = images.shape
        images = images.squeeze(1)  # shape (N, H, W)

        dtn_maps = []
        images = images.to(device)
        for i in tqdm(range(N), desc=f"Processing {split} images"):
            img = images[i]
            dtn_map = dtn_from_sigma(v_h=v_h, sigma_vec=img, img_size=img_size)
            dtn_maps.append(dtn_map.cpu())

        dtn_maps = torch.stack(dtn_maps, dim=0)
        
        dataset[split] = {
            'dtn_map': dtn_maps,
            'media': images  # images already squeezed above
        }

    # refined if we are using the completed mesh in the mesh-data folder.
    save_name = f"data/eit-{args.problem}-dtn-{args.mesh}-128.pt"
    torch.save(dataset, save_name)
    print(f"Saved dataset to {save_name}")

    example_media = dataset['train']['media'][:16].detach().cpu()  # (16, H, W)
    example_dtn = dataset['train']['dtn_map'][:16].detach().cpu()  # (16, n_bdy, n_bdy)

    fig, axes = plt.subplots(4, 8, figsize=(16, 8), dpi=150)
    for i in range(16):
        r = i // 4
        c = i % 4
        ax_media = axes[r, 2 * c]
        ax_dtn = axes[r, 2 * c + 1]

        ax_media.imshow(example_media[i].numpy(), cmap='Blues')
        ax_media.set_title(f"media {i}", fontsize=8)
        ax_media.axis('off')

        ax_dtn.imshow(example_dtn[i].numpy(), cmap='viridis')
        ax_dtn.set_title(f"DtN {i}", fontsize=8)
        ax_dtn.axis('off')

    plt.tight_layout()
    plt.savefig(f"data/eit-{args.problem}-media-dtn-{args.mesh}-examples.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)