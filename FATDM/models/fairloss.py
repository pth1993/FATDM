import torch
import torch.nn as nn
from torch.distributions import Normal


class MMDLoss(nn.Module):
    def __init__(self, device, fair_criteria):
        super(MMDLoss, self).__init__()
        self.device = device
        assert fair_criteria in ['EqOdd', 'EqOpp']
        self.fair_criteria = fair_criteria

    def compute_kernel(self, x, y):
        dim = x.size(1)
        assert dim == y.size(1)
        kernel_input = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def compute_mmd_group(self, x, group):
        unique_groups = group.unique()
        mmd = torch.FloatTensor([0.0]).to(self.device)
        if len(unique_groups) == 1:
            return mmd
        i = 0
        for the_group in unique_groups:
            mmd = mmd + self.compute_mmd(x[group == the_group], x)
            i = i + 1
        return mmd / i

    def forward(self, outputs, labels, group):
        mmd = torch.FloatTensor([0.0]).to(self.device)
        # outputs = F.log_softmax(outputs, dim=1)[:, -1].unsqueeze(1)
        outputs = nn.LogSigmoid()(outputs).unsqueeze(1)
        # outputs = outputs.unsqueeze(1)
        if self.fair_criteria == 'EqOdd':
            unique_labels = [0, 1]
        else:
            unique_labels = [0]
        for the_label in unique_labels:
            if (labels == the_label).sum() > 0:
                mmd = mmd + self.compute_mmd_group(outputs[labels == the_label], group[labels == the_label])
            else:
                print("Skipping regularization due to no samples")
        return mmd


class MeanLoss(nn.Module):
    def __init__(self, device, fair_criteria):
        super(MeanLoss, self).__init__()
        self.device = device
        assert fair_criteria in ['EqOdd', 'EqOpp']
        self.fair_criteria = fair_criteria

    def forward(self, outputs, labels, group):
        result = torch.FloatTensor([0.0]).contiguous().to(self.device)
        outputs = nn.LogSigmoid()(outputs).unsqueeze(1)
        if self.fair_criteria == 'EqOdd':
            unique_labels = [0, 1]
        else:
            unique_labels = [0]
        for the_label in unique_labels:
            if (labels == the_label).sum() > 0:
                result = result + self.compute_mean_gap_group(outputs[labels == the_label], group[labels == the_label])
            else:
                print("Skipping regularization due to no samples")
        return result

    def compute_mean_gap_group(self, outputs, group):
        result = torch.FloatTensor([0.0]).contiguous().to(self.device)
        # unique_groups = group.unique()
        # for i, the_group in enumerate(unique_groups):
        #     result = result + self.compute_mean_gap(outputs[group == the_group], outputs)
        result = result + self.compute_mean_gap(outputs[group == 0], outputs[group == 1])
        return result

    def compute_mean_gap(self, x, y):
        return (x.mean() - y.mean()) ** 2


class CorrLoss(nn.Module):
    def __init__(self, device, fair_criteria):
        super(CorrLoss, self).__init__()
        self.device = device
        assert fair_criteria in ['EqOdd', 'EqOpp']
        self.fair_criteria = fair_criteria

    def forward(self, outputs, labels, group):
        result = torch.FloatTensor([0.0]).to(self.device)
        outputs = nn.LogSigmoid()(outputs)
        group = group.float()
        if self.fair_criteria == 'EqOdd':
            unique_labels = [0, 1]
        else:
            unique_labels = [0]
        for the_label in unique_labels:
            if (labels == the_label).sum() > 0:
                result = result + self.compute_corr_group(outputs[labels == the_label], group[labels == the_label])
            else:
                print("Skipping regularization due to no samples")
        return result

    def compute_corr_group(self, outputs, group):
        result = torch.FloatTensor([0.0]).to(self.device)
        numerator = (outputs * group).mean() - outputs.mean() * group.mean()
        denominator = outputs.std() * group.std()
        result = result + numerator / (denominator + 1e-06)
        return torch.abs(result)


if __name__ == '__main__':
    from scipy.stats import wasserstein_distance
    d1 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    d2 = Normal(torch.tensor([0.0]), torch.tensor([5.0]))
    a = d1.sample([320, 10]).squeeze()
    b = d1.sample([320, 10]).squeeze()
    c = d2.sample([320, 10]).squeeze()
    d = d2.sample([640, 10]).squeeze()
    e = torch.randint(0, 3, (320,)).float()
    f = d1.sample([320])
    mmd_loss = MMDLoss(device='cpu')
    mean_loss = MeanLoss(device='cpu')
    corr_loss = CorrLoss(device='cpu')
    print(wasserstein_distance(a.numpy().flatten(), c.numpy().flatten()))
    print(corr_loss.compute_corr_group(e*0.1+d1.sample([320]), e))
    print(mmd_loss.compute_mmd(a, b), mmd_loss.compute_mmd(a, c), mean_loss.compute_mean_gap(a, b), mean_loss.compute_mean_gap(a, c))
