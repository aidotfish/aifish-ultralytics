import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np


def unnormalize_image(tensor):
    """
    Un-normalizes a tensor image for display.
    Assumes the tensor is normalized with ImageNet's mean and std.
    You might need to adjust these values for your specific dataset.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Ensure tensor is on CPU and in H,W,C format
    tensor = tensor.clone().cpu().numpy().transpose(1, 2, 0)

    # Un-normalize
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor


def visualize_feature_scores(
        pred_scores_after_masking,
        batch,
        item_idx_to_vis,
        gt_bboxes_label_0,
        strides,
        save_path="score_visualization.png"
):
    """
    Visualizes the scores of prediction cells from different feature levels as heatmaps.

    Args:
        pred_scores_after_masking (torch.Tensor): The prediction scores tensor after masking.
                                                  Shape: (batch_size, 8400, num_classes)
        batch (dict): The input batch dictionary, must contain 'img'.
        item_idx_to_vis (int): The index of the item in the batch to visualize.
        gt_bboxes_label_0 (torch.Tensor): The ground truth boxes with label 0 for the specific item.
                                           Shape: (num_boxes, 4)
        strides (list): A list of strides for each feature level, e.g., [8, 16, 32].
        save_path (str): Path to save the output visualization image.
    """
    try:
        # --- 1. Prepare Data ---

        # Get scores for the specific item and take the max score across all classes for visualization
        item_scores = torch.max(pred_scores_after_masking[item_idx_to_vis], dim=-1).values.detach()

        # Un-flatten the scores back into their feature map shapes
        shapes = [(80, 80), (40, 40), (20, 20)]
        split_sizes = [s[0] * s[1] for s in shapes]
        scores_split = torch.split(item_scores, split_sizes)
        score_maps = [s.reshape(shape) for s, shape in zip(scores_split, shapes)]

        # Get the original image and unnormalize it for display
        original_image = unnormalize_image(batch['img'][item_idx_to_vis])

        # --- 2. Create the Plot ---

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Score Visualization for Batch Item {item_idx_to_vis}", fontsize=16)

        # a) Plot Original Image with GT Boxes
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image + GT Boxes (Label 0)")
        axes[0].axis('off')
        for box in gt_bboxes_label_0:
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)

        # b) Plot the 3 score heatmaps
        for i, (score_map, stride) in enumerate(zip(score_maps, strides)):
            ax = axes[i + 1]
            # Use vmin=0, vmax=1 so the color for 0 is consistent
            im = ax.imshow(score_map.cpu().numpy(), cmap='viridis', vmin=0, vmax=1.0)
            ax.set_title(f"Feature Level (Stride {stride})")
            ax.axis('off')

            # Overlay the ground truth boxes, scaled to the feature map's dimensions
            for box in gt_bboxes_label_0:
                # Scale box coordinates down to the feature map grid size
                scaled_box = [coord / stride for coord in box.cpu().numpy()]
                x1, y1, x2, y2 = scaled_box
                width, height = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=1.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for suptitle
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory
        print(f"Saved score visualization to {save_path}")

    except Exception as e:
        print(f"Could not generate visualization. Error: {e}")