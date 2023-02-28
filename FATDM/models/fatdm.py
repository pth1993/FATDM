import torch.nn as nn
import torch.nn.functional as F
import torch
from .fairloss import MeanLoss
from torch.cuda.amp import autocast


class FATDM(nn.Module):
    def __init__(self, feature_extractor, task_classifier, domain_y_transfer, domain_ya_transfer, alpha, device):
        super(FATDM, self).__init__()
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.domain_y_transfer = domain_y_transfer
        self.domain_y_transfer.eval()
        for p in self.domain_y_transfer.parameters():
            p.requires_grad = False
        self.domain_ya_transfer = domain_ya_transfer
        self.domain_ya_transfer.eval()
        for p in self.domain_ya_transfer.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.device = device

    def feature_extractor_func(self, x):
        return self.feature_extractor(x)

    def get_representation(self, feature):
        with autocast():
            return self.feature_extractor(feature)

    def forward(self, feature, y_task, y_domain, c_dim, group, fair_criteria, fair_loss, weight, dataset):
        if dataset in ['train', 'val']:
            with autocast():
                z = self.feature_extractor(feature)
                logits = self.task_classifier(z).squeeze(-1)
                c_loss = F.binary_cross_entropy_with_logits(logits, y_task)
                if self.training:
                    rand_idx = torch.randperm(y_domain.size(0))
                    y_domain_new = y_domain[rand_idx]
                    y_domain_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_onehot.scatter_(1, y_domain[:, None], 1)
                    y_domain_new_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
                    x_new = self.domain_y_transfer(feature, y_domain_onehot, y_domain_new_onehot)
                    z_new = self.feature_extractor(x_new)
                    reg_y = F.mse_loss(z_new, z)

                    rand_idx = torch.randperm(y_domain.size(0))
                    y_domain_new = y_domain[rand_idx]
                    y_domain_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_onehot.scatter_(1, y_domain[:, None], 1)
                    y_domain_new_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
                    x_new = self.domain_ya_transfer(feature, y_domain_onehot, y_domain_new_onehot)
                    z_new = self.feature_extractor(x_new)
                    reg_ya = F.mse_loss(z_new, z)
                else:
                    reg_y = c_loss.new_zeros([1])
                    reg_ya = c_loss.new_zeros([1])
                if fair_loss == 'Mean':
                    f_loss = MeanLoss(self.device, fair_criteria)
                    f_loss = f_loss(logits, y_task, group)
                else:
                    f_loss = torch.zeros(1).to(self.device)
                loss_total = c_loss + self.alpha * (reg_y + reg_ya) + weight * f_loss
            j_loss = c_loss.new_zeros([1])
            return loss_total, c_loss, reg_y + reg_ya, j_loss, f_loss, logits
        else:
            with autocast():
                z = self.feature_extractor(feature)
                logits = self.task_classifier(z).squeeze(-1)
            return logits
