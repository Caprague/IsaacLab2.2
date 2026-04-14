#!/usr/bin/env python
"""Test and visualize box grid pattern ray casting."""

from typing import Literal, Sequence
from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class PatternBaseCfg:
    """Base configuration for a pattern."""

    func: callable = None


class BoxGridPatternCfg(PatternBaseCfg):
    """Configuration for the box grid pattern for ray-casting."""

    resolution: float = 0.5
    size: tuple[float, float, float] = (3.0, 3.0, 4.0)
    directions: Sequence[tuple[float, float, float]] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ordering: Literal["xy", "yx"] = "xy"


def box_grid_pattern(cfg: BoxGridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A 3D box grid pattern for ray casting with multi-face sampling.

    Args:
        cfg: The configuration instance for the box grid pattern.
        device: The device to create the pattern on.

    Returns:
        A tuple containing:
            - ray_starts: Starting positions of rays with shape (total_rays, 3)
            - ray_directions: Direction vectors of rays with shape (total_rays, 3)
    """
    # check valid arguments
    if cfg.ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
    if cfg.resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")

    # box dimensions
    length, width, height = cfg.size

    all_starts = []
    all_directions = []

    # iterate over each direction
    for direction in cfg.directions:
        direction_tensor = torch.tensor(list(direction), device=device, dtype=torch.float32)
        direction_tensor = direction_tensor / torch.norm(direction_tensor)  # normalize

        # find the primary axis (largest absolute component)
        abs_direction = torch.abs(direction_tensor)
        primary_axis = torch.argmax(abs_direction).item()

        # determine the two secondary axes for the 2D grid
        if primary_axis == 0:  # x-direction
            grid_size_1, grid_size_2 = width, height
            grid_dims = [1, 2]  # y, z
            start_offset = -length / 2.0
        elif primary_axis == 1:  # y-direction
            grid_size_1, grid_size_2 = length, height
            grid_dims = [0, 2]  # x, z
            start_offset = -width / 2.0
        else:  # z-direction
            grid_size_1, grid_size_2 = length, width
            grid_dims = [0, 1]  # x, y
            start_offset = -height / 2.0

        # resolve mesh grid indexing
        indexing = cfg.ordering if cfg.ordering == "xy" else "ij"

        # create grid for the sampling face
        grid_1 = torch.arange(
            start=-grid_size_1 / 2, end=grid_size_1 / 2 + 1.0e-9, step=cfg.resolution, device=device
        )
        grid_2 = torch.arange(
            start=-grid_size_2 / 2, end=grid_size_2 / 2 + 1.0e-9, step=cfg.resolution, device=device
        )
        g1, g2 = torch.meshgrid(grid_1, grid_2, indexing=indexing)

        # create ray starts on the face
        num_rays_face = g1.numel()
        ray_starts_face = torch.zeros(num_rays_face, 3, device=device)
        ray_starts_face[:, grid_dims[0]] = g1.flatten()
        ray_starts_face[:, grid_dims[1]] = g2.flatten()
        ray_starts_face[:, primary_axis] = start_offset

        # ray directions are all the same
        ray_directions_face = torch.zeros_like(ray_starts_face)
        ray_directions_face[:, :] = direction_tensor

        all_starts.append(ray_starts_face)
        all_directions.append(ray_directions_face)

    # concatenate all directions
    all_starts = torch.cat(all_starts, dim=0)
    all_directions = torch.cat(all_directions, dim=0)

    return all_starts, all_directions


def visualize_box_grid_pattern():
    """Generate and visualize the box grid pattern."""
    print("Generating box grid pattern...")

    # Create configuration
    cfg = BoxGridPatternCfg()

    # Generate pattern
    ray_starts, ray_directions = box_grid_pattern(cfg, "cpu")

    print(f"Number of rays: {ray_starts.shape[0]}")
    print(f"Ray starts shape: {ray_starts.shape}")
    print(f"Ray starts: {ray_starts}")
    print(f"Ray directions shape: {ray_directions.shape}")
    print(f"Ray directions: {ray_directions}")

    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Colors for different directions
    colors = ['r', 'g', 'b']
    labels = ['X-direction (left->right)', 'Y-direction (back->front)', 'Z-direction (bottom->top)']

    # Plot rays by direction
    for i, direction in enumerate(cfg.directions):
        direction_tensor = torch.tensor(direction, dtype=torch.float32)
        direction_tensor = direction_tensor / torch.norm(direction_tensor)

        # Find rays with this direction
        mask = torch.all(torch.isclose(ray_directions, direction_tensor, atol=1e-5), dim=1)
        starts = ray_starts[mask]

        # Plot start points
        ax.scatter(starts[:, 0].numpy(), starts[:, 1].numpy(), starts[:, 2].numpy(),
                   c=colors[i], marker='o', s=20, alpha=0.6, label=labels[i])

        # Plot ray directions as arrows
        ray_length = 0.3
        # Sample some rays to avoid cluttering
        sample_step = max(1, len(starts) // 30)
        sample_starts = starts[::sample_step]

        # Create arrow direction vectors
        arrow_dirs = direction_tensor.unsqueeze(0).repeat(len(sample_starts), 1)

        # Plot arrows using quiver
        ax.quiver(
            sample_starts[:, 0].numpy(),
            sample_starts[:, 1].numpy(),
            sample_starts[:, 2].numpy(),
            arrow_dirs[:, 0].numpy(),
            arrow_dirs[:, 1].numpy(),
            arrow_dirs[:, 2].numpy(),
            length=ray_length,
            normalize=False,
            color=colors[i],
            alpha=0.4,
            arrow_length_ratio=0.2,
            linewidth=1
        )

    # Draw bounding box
    l, w, h = cfg.size
    corners = [
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2], [l/2, w/2, h/2], [-l/2, w/2, h/2]
    ]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
    ]
    for edge in edges:
        p1, p2 = corners[edge[0]], corners[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'k-', linewidth=1, alpha=0.3)

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Box Grid Pattern (Size: {l}x{w}x{h}m, Resolution: {cfg.resolution}m)\n'
                 f'Total Rays: {ray_starts.shape[0]}')
    ax.legend()

    # Set equal aspect ratio
    max_range = max(l, w, h) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Set view angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_box_grid_pattern()