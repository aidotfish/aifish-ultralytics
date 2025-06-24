import torch
import torch.nn as nn

from ultralytics.utils.metrics import bbox_iou


class BboxLossWithIgnoreZones(nn.Module):
    """
    Criterion class for computing training losses for bounding boxes with ignore zones.
    Extends the BboxLoss class with the ability to ignore detections that fall within specified areas.
    """

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLossWithIgnoreZones module with regularization maximum and DFL settings."""
        # prevent circular import by importing DFLoss here
        from ultralytics.utils.loss import DFLoss
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        ignore_zones: torch.Tensor = None,
        ignore_zones_iou_threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IoU and DFL losses for bounding boxes, excluding boxes in ignore zones.

        Args:
            pred_dist: Predicted distribution
            pred_bboxes: Predicted bounding boxes
            anchor_points: Anchor points
            target_bboxes: Target bounding boxes
            target_scores: Target scores
            target_scores_sum: Sum of target scores
            fg_mask: Foreground mask
            ignore_zones: Ignore zones in [batch_idx, x, y, w, h] format, shape (N, 5)
                          where N is the total number of ignore zones across all batch samples
        
        Returns:
            tuple of (loss_iou, loss_dfl)
        """
        # Start with the original fg_mask
        effective_fg_mask = fg_mask.clone()

        # Apply ignore zones if provided
        if ignore_zones is not None and len(ignore_zones) > 0:
            batch_size = pred_bboxes.shape[0]
            
            # For each item in batch
            for batch_idx in range(batch_size):
                # Get the foreground predictions for this batch item
                batch_fg_mask = fg_mask[batch_idx]
                if not batch_fg_mask.any():
                    continue
                    
                batch_pred_boxes = pred_bboxes[batch_idx][batch_fg_mask]
                
                # Find ignore zones for this batch item
                batch_ignore_zones = ignore_zones[ignore_zones[:, 0] == batch_idx]
                if len(batch_ignore_zones) == 0:
                    continue
                    
                # Convert ignore zones from xywh to xyxy format for IoU calculation
                #TODO: this is still normalized, but should be in pixel coordinates
                ignore_zones_xyxy = torch.zeros_like(batch_ignore_zones[:, 1:])
                ignore_zones_xyxy[:, 0] = batch_ignore_zones[:, 1] - batch_ignore_zones[:, 3] / 2  # x1
                ignore_zones_xyxy[:, 1] = batch_ignore_zones[:, 2] - batch_ignore_zones[:, 4] / 2  # y1
                ignore_zones_xyxy[:, 2] = batch_ignore_zones[:, 1] + batch_ignore_zones[:, 3] / 2  # x2
                ignore_zones_xyxy[:, 3] = batch_ignore_zones[:, 2] + batch_ignore_zones[:, 4] / 2  # y2
                
                # Calculate IoU between each prediction and all ignore zones
                ious = bbox_iou(batch_pred_boxes, ignore_zones_xyxy, xywh=False, CIoU=False)
                
                # Find predictions that overlap with any ignore zone
                # A prediction overlaps if its max IoU with any ignore zone is > ignore_zones_iou_threshold
                max_ious, _ = torch.max(ious, dim=1)
                ignore_mask = max_ious > ignore_zones_iou_threshold
                
                # Update the effective foreground mask
                batch_fg_indices = torch.nonzero(batch_fg_mask).squeeze(-1)
                ignore_indices = batch_fg_indices[ignore_mask]
                
                if len(ignore_indices) > 0:
                    effective_fg_mask[batch_idx, ignore_indices] = False

        # Use the effective fg_mask (with ignore zones applied) for loss calculation
        weight = target_scores.sum(-1)[effective_fg_mask].unsqueeze(-1)
        
        # If no foreground predictions remain after applying ignore zones
        if not effective_fg_mask.any():
            # Return zero loss
            device = pred_dist.device
            return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        
        # Calculate IoU loss with the effective mask
        iou = bbox_iou(pred_bboxes[effective_fg_mask], target_bboxes[effective_fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss with the effective mask
        if self.dfl_loss:
            target_ltrb = self._get_target_ltrb(anchor_points, target_bboxes)
            loss_dfl = self.dfl_loss(
                pred_dist[effective_fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[effective_fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
        
    def _get_target_ltrb(self, anchor_points: torch.Tensor, target_bboxes: torch.Tensor) -> torch.Tensor:
        """
        Helper method to get target box distributions.
        
        This extracts the bbox2dist function call to avoid circular imports.
        """
        from ultralytics.utils.tal import bbox2dist
        return bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
