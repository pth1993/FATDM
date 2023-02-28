import os
import argparse
from torch.backends import cudnn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import ConcatDomainRotatedXrayDataset, ConcatDomainAgeXrayDataset
from models import Generator, Discriminator
import datetime
import time
import random
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.cross_entropy(logit, target)


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


def update_lr(g_optimizer, d_optimizer, g_lr, d_lr):
    """Decay learning rates of the generator and discriminator."""
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = g_lr
    for param_group in d_optimizer.param_groups:
        param_group['lr'] = d_lr


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def train(x_real, c_org, c_trg, label_org, label_trg, D, G, d_optimizer, g_optimizer, scaler_d, scaler_g, current_d_lr,
          current_g_lr, i, classification_loss, start_time, config, device):
    # =================================================================================== #
    #                             2. Train the discriminator                              #
    # =================================================================================== #

    with autocast():
        # Compute loss with real images.
        out_src, out_cls = D(x_real)
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = classification_loss(out_cls, label_org)

        # Compute loss with fake images.
        x_fake = G(x_real, c_org, c_trg)
        out_src, out_cls = D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = D(x_hat)

        weight = torch.ones(out_src.size()).to(device)

    dydx = torch.autograd.grad(outputs=scaler_d.scale(out_src),
                               inputs=x_hat,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    inv_scale = 1. / scaler_d.get_scale()
    dydx = dydx * inv_scale

    with autocast():
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        d_loss_gp = torch.mean((dydx_l2norm - 1) ** 2)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + config.lambda_cls * d_loss_cls + config.lambda_gp * d_loss_gp

    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    scaler_d.scale(d_loss).backward()
    scaler_d.step(d_optimizer)
    scaler_d.update()

    loss = {}
    loss['D/loss_real'] = d_loss_real.item()
    loss['D/loss_fake'] = d_loss_fake.item()
    loss['D/loss_cls'] = d_loss_cls.item()
    loss['D/loss_gp'] = d_loss_gp.item()

    # =================================================================================== #
    #                               3. Train the generator                                #
    # =================================================================================== #

    if (i + 1) % config.n_critic == 0:
        with autocast():
            # Original-to-target domain.
            x_fake = G(x_real, c_org, c_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = - torch.mean(out_src)
            g_loss_cls = classification_loss(out_cls, label_trg)

            # Target-to-original domain.
            x_reconst = G(x_fake, c_trg, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

            # Backward and optimize.
            g_loss = g_loss_fake + config.lambda_rec * g_loss_rec + config.lambda_cls * g_loss_cls
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        scaler_g.scale(g_loss).backward()
        scaler_g.step(g_optimizer)
        scaler_g.update()

        # Logging.
        loss['G/loss_fake'] = g_loss_fake.item()
        loss['G/loss_rec'] = g_loss_rec.item()
        loss['G/loss_cls'] = g_loss_cls.item()

    # =================================================================================== #
    #                                 4. Miscellaneous                                    #
    # =================================================================================== #

    # Print out training information.
    if (i + 1) % config.log_step == 0:
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, config.num_iters)
        for tag, value in loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    # Save model checkpoints.
    if (i + 1) % config.model_save_step == 0:
        G_path = os.path.join(config.model_save_dir, 'stargan_g_%s_%s_%d.ckpt'
                              % (config.test_domain, config.condition, i+1))
        D_path = os.path.join(config.model_save_dir, 'stargan_d_%s_%s_%d.ckpt'
                              % (config.test_domain, config.condition, i+1))
        torch.save(G.state_dict(), G_path)
        torch.save(D.state_dict(), D_path)

        G_path = os.path.join(config.model_save_dir, 'stargan_g_%s_%s_last.ckpt'
                              % (config.test_domain, config.condition))
        D_path = os.path.join(config.model_save_dir, 'stargan_d_%s_%s_last.ckpt'
                              % (config.test_domain, config.condition))
        torch.save(G.state_dict(), G_path)
        torch.save(D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(config.model_save_dir))

    # Decay learning rates.
    if (i + 1) % config.lr_update_step == 0 and (i + 1) > (config.num_iters - config.num_iters_decay):
        current_g_lr -= (config.g_lr / float(config.num_iters_decay))
        current_d_lr -= (config.d_lr / float(config.num_iters_decay))
        update_lr(g_optimizer, d_optimizer, current_g_lr, current_d_lr)
        print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(current_g_lr, current_d_lr))
    return current_g_lr, current_d_lr


def sample_imgs(x_real, c_org, c_trg, G, i, test_domain, condition):
    with autocast():
        x_sample = G(x_real, c_org, c_trg)
    # save_image(denorm(torch.cat((x_real, x_sample), dim=0).data), "output/rotated_xray/stargan/images/%s/%s_%d.png"
    save_image(denorm(torch.cat((x_real, x_sample), dim=0).data), "output/age_xray/stargan/images/%s/%s_%d.png"
               % (test_domain, condition, i), nrow=8)


start_time = time.time()

SEED = 43

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
cudnn.deterministic = True
cudnn.benchmark = True


parser = argparse.ArgumentParser()

# Model configuration.
parser.add_argument('--c_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
parser.add_argument('--image_size', type=int, default=256, help='image resolution')
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

# Training configuration.
parser.add_argument('--test_domain', type=str)
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=300000, help='number of total iterations for training D')
parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

# Directories.
# parser.add_argument('--model_save_dir', type=str, default='saved_model/rotated_xray/stargan')
parser.add_argument('--model_save_dir', type=str, default='saved_model/age_xray/stargan')

# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=10000)
parser.add_argument('--lr_update_step', type=int, default=1000)
parser.add_argument('--gpu', type=str)
parser.add_argument('--condition', type=str, default='none')

config = parser.parse_args()

# metadata_path = 'dataset/metadata/rotated_edema/'
# domain_list = ['0', '15', '30', '45', '60']
# metadata_path_list = [metadata_path + 'Edema_%s.csv' % d for d in domain_list]
# data_path = 'dataset/data/rotated_edema_numpy_1/'

metadata_path = 'dataset/metadata/age_cardiomegaly/'
domain_list = ['0_40', '40_60', '60_80', '80_100']
metadata_path_list = [metadata_path + 'Cardiomegaly_%s.csv' % d for d in domain_list]
data_path = 'dataset/data/age_cardiomegaly_numpy/'

gpu = config.gpu
if gpu != 'osc':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_g_lr = config.g_lr
current_d_lr = config.d_lr

sample_weights = []

train_dataloader_list = []
if config.condition == 'ya':
    for y in [0, 1]:
        # for g in [0, 1, 2, 3]:
        for g in [0, 1]:
            # train_dataset = ConcatDomainRotatedXrayDataset(metadata_path_list, data_path, dataset='train',
            #                                                test_domain=config.test_domain, lb_val=y, g_val=g)
            train_dataset = ConcatDomainAgeXrayDataset(metadata_path_list, data_path, dataset='train',
                                                       test_domain=config.test_domain, lb_val=y, g_val=g)
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=24,
                                          pin_memory=True, drop_last=True)
            train_dataloader_list.append(train_dataloader)
            sample_weights.append(len(train_dataset))
elif config.condition == 'y':
    for y in [0, 1]:
        # train_dataset = ConcatDomainRotatedXrayDataset(metadata_path_list, data_path, dataset='train',
        #                                                test_domain=config.test_domain, lb_val=y, g_val=None)
        train_dataset = ConcatDomainAgeXrayDataset(metadata_path_list, data_path, dataset='train',
                                                   test_domain=config.test_domain, lb_val=y, g_val=None)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=24,
                                      pin_memory=True, drop_last=True)
        train_dataloader_list.append(train_dataloader)
        sample_weights.append(len(train_dataset))
else:
    # train_dataset = ConcatDomainRotatedXrayDataset(metadata_path_list, data_path, dataset='train',
    #                                                test_domain=config.test_domain, lb_val=None, g_val=None)
    train_dataset = ConcatDomainAgeXrayDataset(metadata_path_list, data_path, dataset='train',
                                               test_domain=config.test_domain, lb_val=None, g_val=None)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=24,
                                  pin_memory=True, drop_last=True)
    train_dataloader_list.append(train_dataloader)
    sample_weights.append(len(train_dataset))

G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num)

g_optimizer = torch.optim.Adam(G.parameters(), config.g_lr, betas=(config.beta1, config.beta2))
d_optimizer = torch.optim.Adam(D.parameters(), config.d_lr, betas=(config.beta1, config.beta2))

G.to(device)
D.to(device)

scaler_d = GradScaler()
scaler_g = GradScaler()

start_iters = 0

# Start training.
print('Start training...')
for i in tqdm(range(start_iters, config.num_iters)):

    # =================================================================================== #
    #                             1. Preprocess input data                                #
    # =================================================================================== #

    # Fetch real images and labels.
    idx = random.choices(range(len(train_dataloader_list)), weights=sample_weights)[0]
    train_dataloader = train_dataloader_list[idx]
    try:
        batch = next(data_iter)
    except:
        data_iter = iter(train_dataloader)
        batch = next(data_iter)

    x_real = batch['img']
    lb = batch['lb']
    label_org = batch['d_lb']
    # g = batch['age']
    g = batch['gender']

    # Generate target domain labels randomly.
    rand_idx = torch.randperm(label_org.size(0))
    label_trg = label_org[rand_idx]

    c_org = label2onehot(label_org, config.c_dim)
    c_trg = label2onehot(label_trg, config.c_dim)
    x_real = x_real.to(device)           # Input images.
    c_org = c_org.to(device)             # Original domain labels.
    c_trg = c_trg.to(device)             # Target domain labels.
    label_org = label_org.to(device)     # Labels for computing classification loss.
    label_trg = label_trg.to(device)     # Labels for computing classification loss.
    lb = lb.to(device)
    g = g.to(device)

    current_d_lr, current_g_lr = train(x_real, c_org, c_trg, label_org, label_trg, D, G, d_optimizer,
                                       g_optimizer, scaler_d, scaler_g, current_d_lr, current_g_lr, i,
                                       classification_loss, start_time, config, device)
    if (i+1) % config.sample_step == 0:
        sample_imgs(x_real, c_org, c_trg, G, i+1, config.test_domain, config.condition)
