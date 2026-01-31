import torch
import random
import numpy as np
import tqdm
from math import floor
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def is_valid_circle(center, radius, circles, im_size, min_distance=16, boundary_margin=0):
    image_center = (im_size - 1) / 2.0
    unit_radius = (im_size - 1) / 2.0
    center_distance = ((center[0] - image_center)**2 + (center[1] - image_center)**2)**0.5
    if center_distance + radius + boundary_margin > unit_radius:
        return False

    for existing_circle in circles:
        existing_center, existing_radius = existing_circle
        distance = ((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)**0.5
        if distance < radius + existing_radius + min_distance:
            return False


    return True


def create_circles_dataset(num_samples=5000, im_size=128, problem='ct'):
    # Initialize a tensor to store all images
    dataset = torch.zeros((num_samples, im_size, im_size))

    # you can edit this manually here to change
    # the values of the circles

    # for EIT I used 2 - 5, with h = 5
    # for CT I used 0.4 - 1.4 with h = 20
    if problem == 'eit':
        scattering_indices = np.linspace(2, 5, 5)
    else:
        scattering_indices = np.linspace(0.4, 1.4, 20)

    for sample_idx in tqdm.tqdm(range(num_samples), desc=f"Creating {problem} circles dataset"):
        # for EIT the background should be one, achieved by torch.ones
        # for CT the background should be zero, achieved by torch.zeros
        if problem == 'eit':
            img = torch.ones((im_size, im_size), dtype=torch.float32)
        else:
            img = torch.zeros((im_size, im_size), dtype=torch.float32)

        num_circles = random.randint(1, 5)  # Random number of circles per image
        circles = []

        for _ in range(num_circles):
            while True:
                center_x, center_y = random.randint(floor(im_size*.2), floor(im_size*.8)), random.randint(floor(im_size*.2), floor(im_size*.8))
                radius = random.randint(floor(im_size*.05), floor(im_size*.175))
                if is_valid_circle(
                    (center_x, center_y),
                    radius,
                    circles,
                    im_size=im_size,
                    min_distance=floor(im_size * .005),
                    boundary_margin=floor(im_size * .05),
                ):
                    break

            circles.append(((center_x, center_y), radius))

            # Create a grid of coordinates
            x = torch.arange(im_size).view(-1, 1)
            y = torch.arange(im_size).view(1, -1)

            # Calculate the distance from the center
            distance = (x - center_x)**2 + (y - center_y)**2
            # Create a mask for the circle
            circle_mask = distance <= radius**2

            img[circle_mask] = np.random.choice(scattering_indices, replace=True)

        # Store the image in the dataset
        dataset[sample_idx] = img

    return dataset

parser = argparse.ArgumentParser(description="Generate circles.")
parser.add_argument('--im_size', type=int, default=128)
parser.add_argument('--problem', type=str, default='ct', help='ct or eit')

args = parser.parse_args()
problem = args.problem
assert problem in ['ct', 'eit'], "problem must be either 'ct' or 'eit'"

num_samples=20000
im_size = args.im_size
dataset = create_circles_dataset(num_samples, im_size=im_size, problem=problem)
dataset = dataset.unsqueeze(1) # Add channel dimension

train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
dataset = {
    'train': dataset[:train_size],
    'val': dataset[train_size:train_size + val_size],
    'test': dataset[train_size + val_size:]
}

torch.save(dataset, f"data/{problem}-circles-dataset-{im_size}.pt")
print(f"Dataset saved as {problem}-circles-dataset-{im_size}.pt")

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
plt.savefig(f"data/{problem}-circles-examples-{im_size}.png", bbox_inches='tight', pad_inches=0)
plt.close()