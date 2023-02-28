import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils import ConcatDomainRotatedXrayDataset, ConcatDomainAgeXrayDataset, Metrics
from tqdm import tqdm
from models import FeatureExtractorXray, TaskClassifier, Generator, FATDM
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime
import random
import argparse
import pickle
import os
from torch.cuda.amp import GradScaler


start_time = datetime.now()

parser = argparse.ArgumentParser(description='Adult Training')
parser.add_argument('--train_batch_size', default=128)
parser.add_argument('--test_batch_size', default=128)
parser.add_argument('--max_epoch', default=10)
parser.add_argument('--test_domain')
parser.add_argument('--fair_criteria')
parser.add_argument('--fair_loss')
parser.add_argument('--fair_weight')
parser.add_argument('--model_name')
parser.add_argument('--gpu')
parser.add_argument('--seed', default=43)


args = parser.parse_args()

SEED = int(args.seed)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

max_epoch = args.max_epoch
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
test_domain = args.test_domain
model_name = args.model_name
fair_criteria = args.fair_criteria
fair_loss = args.fair_loss
if fair_loss == 'None':
    fair_loss = None
fair_weight = float(args.fair_weight)
# metadata_path = 'dataset/metadata/rotated_edema/'
# domain_list = ['0', '15', '30', '45', '60']
# metadata_path_list = [metadata_path + 'Edema_%s.csv' % d for d in domain_list]
# data_path = 'dataset/data/rotated_edema_numpy_1/'

metadata_path = 'dataset/metadata/age_cardiomegaly/'
domain_list = ['0_40', '40_60', '60_80', '80_100']
metadata_path_list = [metadata_path + 'Cardiomegaly_%s.csv' % d for d in domain_list]
data_path = 'dataset/data/age_cardiomegaly_numpy/'

gpu = args.gpu
if gpu != 'osc':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if fair_loss:
#     out_dir = 'saved_model/rotated_xray_multiple/%s_%s_%s_%s_%s_%.4f.ckpt' \
#               % (model_name, args.seed, test_domain, fair_criteria, fair_loss, fair_weight)
# else:
#     out_dir = 'saved_model/rotated_xray_multiple/%s_%s_%s_%s_None_%.4f.ckpt' \
#               % (model_name, args.seed, test_domain, fair_criteria, fair_weight)

out_dir = 'saved_model/age_xray/%s_%s_%s_%s_%s_%.4f.ckpt' \
          % (model_name, args.seed, test_domain, fair_criteria, fair_loss, fair_weight)

# group_map = {0: '< 40', 1: '40 -> 60', 2: '60 -> 80', 3: '> 80'}
group_map = {0: 'F', 1: 'M'}
metric = Metrics(group_map, fair_criteria)

# train_dataset = ConcatDomainRotatedXrayDataset(metadata_path_list, data_path, dataset='train', test_domain=test_domain,
#                                                lb_val=None, g_val=None)
# val_dataset = ConcatDomainRotatedXrayDataset(metadata_path_list, data_path, dataset='val', test_domain=test_domain,
#                                              lb_val=None, g_val=None)
# test_dataset = ConcatDomainRotatedXrayDataset(metadata_path_list, data_path, dataset='test', test_domain=test_domain,
#                                               lb_val=None, g_val=None)

train_dataset = ConcatDomainAgeXrayDataset(metadata_path_list, data_path, dataset='train', test_domain=test_domain,
                                           lb_val=None, g_val=None)
val_dataset = ConcatDomainAgeXrayDataset(metadata_path_list, data_path, dataset='val', test_domain=test_domain,
                                         lb_val=None, g_val=None)
test_dataset = ConcatDomainAgeXrayDataset(metadata_path_list, data_path, dataset='test', test_domain=test_domain,
                                          lb_val=None, g_val=None)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

feature_extractor = FeatureExtractorXray(out_feature=512)
task_classifier = TaskClassifier(in_feature=512)
domain_y_transfer = Generator(conv_dim=64, c_dim=len(domain_list)-1, repeat_num=6, img_channels=1)
domain_y_transfer.load_state_dict(torch.load('saved_model/age_xray/stargan/stargan_g_%s_y_last.ckpt'
                                             % test_domain, map_location=device))
domain_ya_transfer = Generator(conv_dim=64, c_dim=len(domain_list)-1, repeat_num=6, img_channels=1)
domain_ya_transfer.load_state_dict(torch.load('saved_model/age_xray/stargan/stargan_g_%s_ya_last.ckpt'
                                              % test_domain, map_location=device))

scaler = GradScaler()

model = FATDM(feature_extractor=feature_extractor, task_classifier=task_classifier, domain_y_transfer=domain_y_transfer,
              domain_ya_transfer=domain_ya_transfer, alpha=1., device=device)

# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=0.001,
#                             momentum=0.9)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,
                             weight_decay=0, betas=(0.9, 0.999))

model = model.to(device)

# training
best_val_loss = float('inf')
best_val_auc = float('-inf')
val_auc_list = []
test_auc_list = []
val_loss_list = []
for epoch in range(max_epoch):
    print("Iteration %d:" % (epoch + 1))
    model.train()
    epoch_loss = 0
    epoch_c_loss = 0
    epoch_r_loss = 0
    epoch_j_loss = 0
    epoch_f_loss = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        img = batch['img'].to(device)
        lb = batch['lb'].to(device)
        d_lb = batch['d_lb'].to(device)
        # g = batch['age'].to(device)
        g = batch['gender'].to(device)
        optimizer.zero_grad()
        loss, c_loss, r_loss, j_loss, f_loss, predict = \
            model(img, lb, d_lb, len(domain_list)-1, g, fair_criteria, fair_loss, fair_weight, 'train')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_c_loss += c_loss.item()
        epoch_r_loss += r_loss.item()
        epoch_j_loss += j_loss.item()
        epoch_f_loss += f_loss.item()
    print('Train loss: %.4f - C loss: %.4f - R loss: %.4f - J loss: %.4f - F loss: %.4f'
          % (epoch_loss / (i + 1), epoch_c_loss / (i + 1), epoch_r_loss / (i + 1), epoch_j_loss / (i + 1), epoch_f_loss / (i + 1)))

    model.eval()
    model.domain_y_transfer.train()
    model.domain_ya_transfer.train()
    epoch_loss = 0
    epoch_c_loss = 0
    epoch_r_loss = 0
    epoch_j_loss = 0
    epoch_f_loss = 0
    with torch.no_grad():
        predict_list = np.empty(0)
        lb_list = np.empty(0)
        group_list = np.empty(0)
        for i, batch in enumerate(tqdm(val_dataloader)):
            img = batch['img'].to(device)
            lb = batch['lb'].to(device)
            d_lb = batch['d_lb'].to(device)
            # g = batch['age'].to(device)
            g = batch['gender'].to(device)
            loss, c_loss, r_loss, j_loss, f_loss, predict = \
                model(img, lb, d_lb, len(domain_list)-1, g, fair_criteria, fair_loss, fair_weight, 'val')
            predict_list = np.concatenate((predict_list, predict.squeeze(-1).cpu().numpy()), axis=0)
            lb_list = np.concatenate((lb_list, lb.cpu().numpy()), axis=0)
            group_list = np.concatenate((group_list, g.cpu().numpy()), axis=0)
            epoch_loss += loss.item()
            epoch_c_loss += c_loss.item()
            epoch_r_loss += r_loss.item()
            epoch_j_loss += j_loss.item()
            epoch_f_loss += f_loss.item()
        auc_score = roc_auc_score(lb_list, predict_list)
        val_auc_list.append(auc_score)
        val_loss_list.append(epoch_loss / (i + 1))
        fpr, tpr, thresholds = roc_curve(lb_list, predict_list)
        t = thresholds[np.argmax(tpr - fpr)]
        predict_list_binary = (predict_list > t) * 1
        output = {'soft_predict': predict_list, 'label': lb_list, 'group': group_list,
                  'hard_predict': predict_list_binary}
        score = metric.calculate_metrics(output)
        a, b, _ = score['fairness']['all']
    print('Val loss: %.4f - C loss: %.4f - R loss: %.4f - J loss: %.4f - F loss: %.4f - AUC: %.4f - F(P)PR gap: %.4f - Mean: %.4f'
          % (epoch_loss / (i + 1), epoch_c_loss / (i + 1),  epoch_r_loss / (i + 1), epoch_j_loss / (i + 1), epoch_f_loss / (i + 1), auc_score,
             a, b))

    if best_val_auc < auc_score:
        best_val_auc = auc_score
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, out_dir)

    model.eval()
    model.domain_y_transfer.train()
    model.domain_ya_transfer.train()
    with torch.no_grad():
        predict_list = np.empty(0)
        lb_list = np.empty(0)
        group_list = np.empty(0)
        for i, batch in enumerate(tqdm(test_dataloader)):
            img = batch['img'].to(device)
            lb = batch['lb'].to(device)
            # g = batch['age'].to(device)
            g = batch['gender'].to(device)
            predict = model(img, lb, None, len(domain_list)-1, g, fair_criteria, fair_loss, fair_weight, 'test')
            predict_list = np.concatenate((predict_list, predict.squeeze(-1).cpu().numpy()), axis=0)
            lb_list = np.concatenate((lb_list, lb.cpu().numpy()), axis=0)
            group_list = np.concatenate((group_list, g.cpu().numpy()), axis=0)
        auc_score = roc_auc_score(lb_list, predict_list)
        test_auc_list.append(auc_score)
        predict_list_binary = (predict_list > t) * 1
        output = {'soft_predict': predict_list, 'label': lb_list, 'group': group_list,
                  'hard_predict': predict_list_binary}
        score = metric.calculate_metrics(output)
        a, b, _ = score['fairness']['all']
    print('Test AUC: %.4f - F(P)PR gap: %.4f - Mean: %.4f' % (auc_score, a, b))

best_val_epoch = np.argmax(val_auc_list)

end_time = datetime.now()

print('Running time: %s' % (end_time - start_time))
print("AUC score on val set by epoch %d (best val epoch): %.4f" % (best_val_epoch + 1, val_auc_list[best_val_epoch]))
print("AUC score on test set by epoch %d (best val epoch): %.4f" % (best_val_epoch + 1, test_auc_list[best_val_epoch]))

# inference
checkpoint = torch.load(out_dir, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
model.domain_y_transfer.train()
model.domain_ya_transfer.train()
with torch.no_grad():
    predict_list = np.empty(0)
    lb_list = np.empty(0)
    for i, batch in enumerate(val_dataloader):
        img = batch['img'].to(device)
        lb = batch['lb'].to(device)
        d_lb = batch['d_lb'].to(device)
        # g = batch['age'].to(device)
        g = batch['gender'].to(device)
        loss, c_loss, r_loss, j_loss, f_loss, predict = model(img, lb, d_lb, len(domain_list)-1, g, fair_criteria, fair_loss, fair_weight, 'val')
        predict_list = np.concatenate((predict_list, predict.squeeze(-1).cpu().numpy()), axis=0)
        lb_list = np.concatenate((lb_list, lb.cpu().numpy()), axis=0)
    fpr, tpr, thresholds = roc_curve(lb_list, predict_list)
    t = thresholds[np.argmax(tpr - fpr)]

with torch.no_grad():
    predict_list = np.empty(0)
    lb_list = np.empty(0)
    group_list = np.empty(0)
    for i, batch in enumerate(test_dataloader):
        img = batch['img'].to(device)
        lb = batch['lb'].to(device)
        # g = batch['age'].to(device)
        g = batch['gender'].to(device)
        predict = model(img, lb, None, len(domain_list)-1, g, fair_criteria, fair_loss, fair_weight, 'test')
        predict_list = np.concatenate((predict_list, predict.squeeze(-1).cpu().numpy()), axis=0)
        lb_list = np.concatenate((lb_list, lb.cpu().numpy()), axis=0)
        group_list = np.concatenate((group_list, g.cpu().numpy()), axis=0)
    predict_list_binary = (predict_list > t) * 1
    output = {'soft_predict': predict_list, 'label': lb_list, 'group': group_list, 'hard_predict': predict_list_binary}
metric = Metrics(group_map, fair_criteria)
score = metric.calculate_metrics(output)
print('Evaluation')
print('AUROC: %.6f - AUPRC: %.6f - CE: %.6f - Acc: %.6f - F1: %.6f' % tuple(score['performance']['all']))
print('F(T)PR gap: %.6f - Mean: %.6f - EMD: %.6f' % tuple(score['fairness']['all']))
# if fair_loss:
#     out_dir = 'output/rotated_xray_multiple/score_%s_%s_%s_%s_%s_%.4f.pkl' \
#               % (model_name, args.seed, test_domain, fair_criteria, fair_loss, fair_weight)
# else:
#     out_dir = 'output/rotated_xray_multiple/score_%s_%s_%s_%s_None_%.4f.pkl' \
#               % (model_name, args.seed, test_domain, fair_criteria, fair_weight)
if fair_loss:
    out_dir = 'output/age_xray/score_%s_%s_%s_%s_%s_%.4f.pkl' \
              % (model_name, args.seed, test_domain, fair_criteria, fair_loss, fair_weight)
with open(out_dir, 'wb') as f:
    pickle.dump(score, f)
