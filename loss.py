import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: float = 0.0,
                 alpha: float = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position=True) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(target, num_classes=logits_size).float() if self.index_label_position else target.float()
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0 :
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:, label_idx] >= threshold) | pos_example.view(-1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)


class FocalLoss(_Loss):
    """
    Focal loss(https://arxiv.org/pdf/1708.02002.pdf)
    Shape:
        - input: (N, C)
        - target: (N)
        - Output: Scalar loss
    Examples:
        >>> loss = FocalLoss(gamma=2, alpha=[1.0]*7)
        >>> input = torch.randn(3, 7, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(7)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, gamma=0, alpha: List[float] = None, reduction="none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(self.alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(self.alpha,list): self.alpha = torch.FloatTensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N, C]
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)
        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        if self.alpha is not None:
            # [N] at[i] = alpha[target[i]]
            at = self.alpha.gather(0, target.squeeze(-1))
            logpt = logpt * at

        loss = -1 * ((1 - pt) ** self.gamma) * logpt
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    @staticmethod
    def convert_binary_pred_to_two_dimension(x, is_logits=True):
        """
        Args:
            x: (*): (log) prob of some instance has label 1
            is_logits: if True, x represents log prob; otherwhise presents prob
        Returns:
            y: (*, 2), where y[*, 1] == log prob of some instance has label 0,
                             y[*, 0] = log prob of some instance has label 1
        """
        probs = torch.sigmoid(x) if is_logits else x
        probs = probs.unsqueeze(-1)
        probs = torch.cat([1-probs, probs], dim=-1)
        logprob = torch.log(probs+1e-4)  # 1e-4 to prevent being rounded to 0 in fp16
        return logprob

    def __str__(self):
        return f"Focal Loss gamma:{self.gamma}"

    def __repr__(self):
        return str(self)


class PolyLoss(_Loss):
    def __init__(self,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 with_logits: bool = True,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=0)
        self.with_logits = with_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        logits_size = input.shape[-1]
        self.ce_loss = self.cross_entropy(input, target)
        
        if len(input.shape) != len(target.shape):
          target = F.one_hot(target, num_classes=logits_size).float()
        if self.with_logits:
          input = torch.nn.Softmax(dim=1)(input) if self.with_logits else input
        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return polyl

class PolyFocalLoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 alpha: List[float] = None,
                 epsilon: float = 1.0,
                 gamma: float = 0.5,
                 with_logits: bool = True,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(self.alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(self.alpha,list): self.alpha = torch.FloatTensor(alpha)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N, C]
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)
        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        if self.alpha is not None:
            # [N] at[i] = alpha[target[i]]
            at = self.alpha.gather(0, target.squeeze(-1))
            logpt = logpt * at

        self.focal_loss = -1 * (1 - pt) ** self.gamma * logpt

        poly_loss = self.focal_loss + self.epsilon * ((1 - pt) ** (self.gamma + 1))

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return polyl

class ComboLoss(_Loss):
    def __init__(self, 
                 reduction: str = 'mean',
                 alpha: float = 0.75,
                 ce_ratio: float = 0.5,
                 epsilon: float = 1e-9,
                 smooth: int = 1):
        super(ComboLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.epsilon = epsilon
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def forward(self, input, target):
        
        #flatten label and prediction tensors
        input = input.view(-1)
        target = target.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (input * target).sum()    
        dice = (2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)
        
        input = torch.clamp(input, self.epsilon, 1.0 - self.epsilon)       
        ce_loss = - (self.alpha * ((target * torch.log(input)) + ((1 - self.alpha) * (1.0 - target) * torch.log(1.0 - input))))
        if self.reduction == 'mean':
            ce_loss = torch.mean(ce_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            ce_loss = torch.sum(ce_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            ce_loss = ce_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        combo = (self.ce_ratio * ce_loss) - ((1 - self.ce_ratio) * dice)
        
        return combo
