import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np


# unnormalize_image function remains the same...
def unnormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor


def visualize_feature_scores(
        pred_scores_after_masking,
        batch,
        item_idx_to_vis,
        gt_bboxes_to_vis,
        label_to_mask,
        strides,
        save_path="score_visualization.png"
):
    """
    Visualizes the scores of prediction cells from different feature levels as heatmaps.
    """
    try:
        # --- 1. Prepare Data ---

        pred_probs = torch.sigmoid(pred_scores_after_masking)

        # Now get the max probability across all classes for each cell
        item_scores = torch.max(pred_probs[item_idx_to_vis], dim=-1).values.detach()

        # Dynamically get feature map shapes from the `feats` tensor list
        # Assumes 'feats' is part of the batch dict
        shapes = [(80, 80), (40, 40), (20, 20)]
        split_sizes = [s[0] * s[1] for s in shapes]

        if sum(split_sizes) != item_scores.shape[0]:
            print(f"Shape mismatch: Sum of feature maps {sum(split_sizes)} != scores {item_scores.shape[0]}")
            return

        scores_split = torch.split(item_scores, split_sizes)
        score_maps = [s.reshape(shape) for s, shape in zip(scores_split, shapes)]

        original_image = unnormalize_image(batch['img'][item_idx_to_vis])

        # --- 2. Create the Plot ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Score Visualization for Batch Item {item_idx_to_vis}", fontsize=16)

        # Plot Original Image
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image + GT Boxes (Label {label_to_mask})")
        axes[0].axis('off')
        for box in gt_bboxes_to_vis:
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)

        # Plot the 3 score heatmaps
        for i, (score_map, stride) in enumerate(zip(score_maps, strides)):
            ax = axes[i + 1]
            im = ax.imshow(score_map.cpu().numpy(), cmap='viridis', vmin=0, vmax=1.0)
            ax.set_title(f"Feature Level (Stride {stride})")
            ax.axis('off')

            # Add colorbar to each heatmap
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Score Value')

            for box in gt_bboxes_to_vis:
                scaled_box = [coord / stride for coord in box.cpu().numpy()]
                x1, y1, x2, y2 = scaled_box
                width, height = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=1.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved score visualization to {save_path}")

    except Exception as e:
        print(f"Could not generate visualization. Error: {e}")