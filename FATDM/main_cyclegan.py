import argparse
import os
import itertools
import time
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from cyclegan_utils.models import *
from cyclegan_utils.utils import *
from cyclegan_utils.datareader import *
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_step", type=int, default=1000, help="interval between saving generator outputs")
parser.add_argument("--model_save_step", type=int, default=5000, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--domain_a", help="domain A")
parser.add_argument("--domain_b", help="domain B")
parser.add_argument('--gpu', type=str)
parser.add_argument('--condition', type=str, default='none')
parser.add_argument('--num_iters', type=int, default=50000, help='number of total iterations for training D')

opt = parser.parse_args()

gpu = opt.gpu
if gpu != 'osc':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create sample and checkpoint directories
os.makedirs("images", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

metadata_path_a = '../fairness-chest-xray-image/dataset/metadata/age_cardiomegaly/Cardiomegaly_%s.csv' % opt.domain_a
metadata_path_b = '../fairness-chest-xray-image/dataset/metadata/age_cardiomegaly/Cardiomegaly_%s.csv' % opt.domain_b
data_path = '../fairness-chest-xray-image/dataset/data/age_cardiomegaly_numpy/'

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
G_AB = G_AB.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)
D_B = D_B.to(device)


G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

scaler_da = GradScaler()
scaler_db = GradScaler()
scaler_g = GradScaler()

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

train_dataloader_list_a = []
train_dataloader_list_b = []
if opt.condition == 'ya':
    for y in [0, 1]:
        for g in [0, 1]:
            train_dataset_a = SingleDomainAgeXrayDataset(metadata_path_a, data_path, domain=opt.domain_a, lb_val=y,
                                                         g_val=g)
            train_dataloader_a = DataLoader(train_dataset_a, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                            pin_memory=True, drop_last=True)
            train_dataloader_list_a.append(train_dataloader_a)

            train_dataset_b = SingleDomainAgeXrayDataset(metadata_path_b, data_path, domain=opt.domain_b, lb_val=y,
                                                         g_val=g)
            train_dataloader_b = DataLoader(train_dataset_b, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                            pin_memory=True, drop_last=True)
            train_dataloader_list_b.append(train_dataloader_b)
elif opt.condition == 'y':
    for y in [0, 1]:
        train_dataset_a = SingleDomainAgeXrayDataset(metadata_path_a, data_path, domain=opt.domain_a, lb_val=y,
                                                     g_val=None)
        train_dataloader_a = DataLoader(train_dataset_a, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                        pin_memory=True, drop_last=True)
        train_dataloader_list_a.append(train_dataloader_a)

        train_dataset_b = SingleDomainAgeXrayDataset(metadata_path_b, data_path, domain=opt.domain_b, lb_val=y,
                                                     g_val=None)
        train_dataloader_b = DataLoader(train_dataset_b, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                        pin_memory=True, drop_last=True)
        train_dataloader_list_b.append(train_dataloader_b)
else:
    pass


def sample_images(i, real_A, real_B, domain_a, domain_b):
    """Saves a generated sample from the test set"""
    G_AB.eval()
    G_BA.eval()
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=8, normalize=True)
    real_B = make_grid(real_B, nrow=8, normalize=True)
    fake_A = make_grid(fake_A, nrow=8, normalize=True)
    fake_B = make_grid(fake_B, nrow=8, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s_%s_%d.png" % (domain_a, domain_b, i), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
num_batch_list = [min(len(a), len(b)) for a, b in zip(train_dataloader_list_a, train_dataloader_list_b)]
train_iter_list_a = [iter(a) for a in train_dataloader_list_a]
train_iter_list_b = [iter(a) for a in train_dataloader_list_b]

start_iters = 0

# Start training.
print('Start training...')
for i in tqdm(range(start_iters, opt.num_iters)):
    idx = random.choices(range(len(train_dataloader_list_a)))[0]
    try:
        batch_a = next(train_iter_list_a[idx])
    except:
        train_iter_list_a[idx] = iter(train_dataloader_list_a[idx])
        batch_a = next(train_iter_list_a[idx])

    try:
        batch_b = next(train_iter_list_b[idx])
    except:
        train_iter_list_b[idx] = iter(train_dataloader_list_b[idx])
        batch_b = next(train_iter_list_b[idx])

    # Set model input
    real_A = batch_a['img'].to(device)
    real_B = batch_b['img'].to(device)
    valid = torch.from_numpy(np.ones((real_A.size(0), *D_A.output_shape), dtype=np.float32)).to(device)
    fake = torch.from_numpy(np.zeros((real_A.size(0), *D_A.output_shape), dtype=np.float32)).to(device)


    # ------------------
    #  Train Generators
    # ------------------

    G_AB.train()
    G_BA.train()

    optimizer_G.zero_grad()

    with autocast():

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

    # loss_G.backward()
    # optimizer_G.step()

    scaler_g.scale(loss_G).backward()
    scaler_g.step(optimizer_G)
    scaler_g.update()

    # -----------------------
    #  Train Discriminator A
    # -----------------------

    optimizer_D_A.zero_grad()

    with autocast():

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

    # loss_D_A.backward()
    # optimizer_D_A.step()

    scaler_da.scale(loss_D_A).backward()
    scaler_da.step(optimizer_D_A)
    scaler_da.update()

    # -----------------------
    #  Train Discriminator B
    # -----------------------

    optimizer_D_B.zero_grad()

    with autocast():

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

    # loss_D_B.backward()
    # optimizer_D_B.step()

    scaler_db.scale(loss_D_B).backward()
    scaler_db.step(optimizer_D_B)
    scaler_db.update()

    loss_D = (loss_D_A + loss_D_B) / 2

    if (i+1) % 100 == 0:

        print(
            "\r[Iter %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f]"
            % (
                i+1,
                opt.num_iters,
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
            )
        )

    if (i + 1) % opt.model_save_step == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_model/cyclegan_G_%s_%s_%s_%d.ckpt" % (opt.condition, opt.domain_a, opt.domain_b, i+1))
        torch.save(G_BA.state_dict(), "saved_model/cyclegan_G_%s_%s_%s_%d.ckpt" % (opt.condition, opt.domain_b, opt.domain_a, i+1))
        torch.save(D_A.state_dict(), "saved_model/cyclegan_D_%s_%s_%d.ckpt" % (opt.condition, opt.domain_a, i+1))
        torch.save(D_B.state_dict(), "saved_model/cyclegan_D_%s_%s_%d.ckpt" % (opt.condition, opt.domain_b, i+1))

    if (i+1) % opt.sample_step == 0:
        sample_images(i+1, real_A, real_B, opt.domain_a, opt.domain_b)
