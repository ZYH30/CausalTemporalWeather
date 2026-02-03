import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output  # 反转梯度


def update_epoch_loss(epoch_loss, loss_result):
    """更新epoch累计损失"""
    epoch_loss['total'] += loss_result['total_loss'].item()
    epoch_loss['x'] += loss_result['pre_y_loss'].item()
    epoch_loss['y'] += loss_result['y_loss'].item()
    return epoch_loss


class AdversarialLoss:
    def __init__(self, ad_weight=1e-3, nor_weight=1, loss_fn = 'MSE'):
        """对抗训练损失计算器 (使用MSE损失)"""
        # self.criterion = nn.MSELoss() # nn.HuberLoss(delta = 1.0)
        if loss_fn == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'huber':
            self.criterion = nn.HuberLoss(delta = 1.0)
        else:
            print('unknown loss_fn, set MSE')
            self.criterion = nn.MSELoss()
            
        self.ad_weight = ad_weight
        self.nor_weight = nor_weight

    def compute_losses(self, outputs, labels, pre_labels, pred_pre_y, is_adversarial):
        """
        计算当前训练步骤的损失

        参数:
            outputs: 主模型输出
            targets: 真实标签
            pred_x: 对抗模型预测
            label_x: 对抗标签
            is_adversarial: 是否对抗训练阶段
        """
        # 基础损失计算
        y_loss = self.criterion(outputs, labels)

        if is_adversarial:
            pre_y_loss = self.criterion(pred_pre_y, pre_labels)
            return {
                'total_loss': pre_y_loss,
                'pre_y_loss': pre_y_loss,
                'y_loss': y_loss,
                'update_x': True
            }
        else:
            pred_pre_y = GradientReversal.apply(pred_pre_y)
            pre_y_loss = self.criterion(pred_pre_y, pre_labels)
            return {
                'total_loss': self.nor_weight * y_loss + self.ad_weight * pre_y_loss,
                'pre_y_loss': pre_y_loss,
                'y_loss': y_loss,
                'update_x': False
            }