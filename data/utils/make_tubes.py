import torch
import random
import numpy as np
import tqdm
from math import floor, cos, sin, radians
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def is_valid_tube(center, length, width, angle, tubes, im_size, min_distance=8, boundary_margin=0):
    """
    Check if a tube (rotated rectangle) is valid:
    - Within unit circle boundary
    - No overlap with existing tubes
    """
    image_center = (im_size - 1) / 2.0
    unit_radius = (im_size - 1) / 2.0
    
    # Check if tube corners are within unit circle
    # Calculate the four corners of the rotated rectangle
    angle_rad = radians(angle)
    cos_a, sin_a = cos(angle_rad), sin(angle_rad)
    
    # Half dimensions
    half_length = length / 2
    half_width = width / 2
    
    # Corner offsets from center (before rotation)
    corners = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width)
    ]
    
    # Rotate and translate corners
    for dx, dy in corners:
        rotated_x = dx * cos_a - dy * sin_a + center[0]
        rotated_y = dx * sin_a + dy * cos_a + center[1]
        
        # Check distance from image center
        dist_from_center = ((rotated_x - image_center)**2 + (rotated_y - image_center)**2)**0.5
        if dist_from_center + boundary_margin > unit_radius:
            return False
    
    # Check overlap with existing tubes
    for existing_tube in tubes:
        existing_center, existing_length, existing_width, existing_angle = existing_tube
        
        # Simple distance check between centers as a first approximation
        center_dist = ((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)**0.5
        
        # Minimum distance needed to avoid overlap (conservative estimate)
        min_sep = (length + existing_length) / 2 + (width + existing_width) / 2 + min_distance
        
        if center_dist < min_sep:
            return False
    
    return True


def create_tube_mask(center, length, width, angle, im_size):
    """
    Create a binary mask for a tube (rotated rectangle)
    """
    img = torch.zeros((im_size, im_size), dtype=torch.float32)
    
    # Create coordinate grids
    x = torch.arange(im_size).float()
    y = torch.arange(im_size).float()
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Translate to tube center
    xx_centered = xx - center[0]
    yy_centered = yy - center[1]
    
    # Rotate coordinates to align with tube
    angle_rad = radians(angle)
    cos_a, sin_a = cos(angle_rad), sin(angle_rad)
    
    xx_rotated = xx_centered * cos_a + yy_centered * sin_a
    yy_rotated = -xx_centered * sin_a + yy_centered * cos_a
    
    # Create tube mask (rotated rectangle)
    mask = (torch.abs(xx_rotated) <= length/2) & (torch.abs(yy_rotated) <= width/2)
    
    return mask


def create_tubes_dataset(num_samples=5000, im_size=128, problem='ct'):
    """
    Create dataset with non-overlapping tubes inside unit circle
    """
    dataset = torch.zeros((num_samples, im_size, im_size))

    # you can edit this manually here to change
    # the values of the circles

    # for EIT I used 2 - 5, with h = 5
    # for CT I used 0.4 - 1.4 with h = 20
    if problem == 'eit':
        scattering_indices = np.linspace(2, 5, 5)
    else:
        scattering_indices = np.linspace(0.4, 1.4, 20)

    for sample_idx in tqdm.tqdm(range(num_samples), desc=f"Creating {problem} tubes dataset"):
        # for EIT the background should be one, achieved by torch.ones
        # for CT the background should be zero, achieved by torch.zeros
        if problem == 'eit':
            img = torch.ones((im_size, im_size), dtype=torch.float32)
        else:
            img = torch.zeros((im_size, im_size), dtype=torch.float32)

        num_tubes = random.randint(2, 6)  # Random number of tubes per image
        tubes = []

        for _ in range(num_tubes):
            attempts = 0
            while attempts < 100:  # Limit attempts to avoid infinite loops
                # Random position within central region
                center_x = random.randint(floor(im_size*0.3), floor(im_size*0.7))
                center_y = random.randint(floor(im_size*0.3), floor(im_size*0.7))
                
                # Random dimensions (elongated shapes)
                length = random.randint(floor(im_size*0.25), floor(im_size*0.6))
                width = random.randint(floor(im_size*0.08), floor(im_size*0.15))
                
                # Random angle
                angle = random.uniform(0, 180)
                
                if is_valid_tube(
                    (center_x, center_y),
                    length,
                    width,
                    angle,
                    tubes,
                    im_size=im_size,
                    min_distance=floor(im_size * 0.02),
                    boundary_margin=floor(im_size * 0.05),
                ):
                    tubes.append(((center_x, center_y), length, width, angle))
                    
                    # Create tube mask and apply scattering value
                    tube_mask = create_tube_mask((center_x, center_y), length, width, angle, im_size)
                    img[tube_mask] = np.random.choice(scattering_indices, replace=True)
                    break
                
                attempts += 1

        # Store the image in the dataset
        dataset[sample_idx] = img

    return dataset

parser = argparse.ArgumentParser(description="Generate tubes.")
parser.add_argument('--im_size', type=int, default=128)
parser.add_argument('--problem', type=str, default='ct', help='ct or eit')

args = parser.parse_args()
problem = args.problem
assert problem in ['ct', 'eit'], "problem must be either 'ct' or 'eit'"

num_samples=20000
im_size = args.im_size
dataset = create_tubes_dataset(num_samples, im_size=im_size, problem=problem)
dataset = dataset.unsqueeze(1) # Add channel dimension

train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
dataset = {
    'train': dataset[:train_size],
    'val': dataset[train_size:train_size + val_size],
    'test': dataset[train_size + val_size:]
}

torch.save(dataset, f"data/{problem}-tubes-dataset-{im_size}.pt")
print(f"Dataset saved as {problem}-tubes-dataset-{im_size}.pt")

# create a mosaic of example images
example_images = dataset['train'][:16]
example_images = example_images[:, 0]  # (16, H, W)
example_images = example_images.view(4, 4, im_size, im_size)
example_images = example_images.permute(0, 2, 1, 3).reshape(4 * im_size, 4 * im_size)

plt.figure(figsize=(6, 6), dpi=150)
plt.imshow(example_images.cpu().numpy(), cmap='Blues')
ax = plt.gca()
image_center = (im_size - 1) / 2.0
unit_radius = (im_size - 1) / 2.0
for row in range(4):
    for col in range(4):
        cx = col * im_size + image_center
        cy = row * im_size + image_center
        ax.add_patch(Circle((cx, cy), unit_radius, fill=False, edgecolor='black', linewidth=1.0))
plt.axis('off')
plt.tight_layout()
plt.savefig(f"data/{problem}-tubes-examples-{im_size}.png", bbox_inches='tight', pad_inches=0)
plt.close()