import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
import datetime

from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Primary device
torch.autograd.set_detect_anomaly(True)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.dataset_Mayo import ImageDataset_Mayo
from model.classifier import Classifier  # noqa
from model.classifier_dis import Classifier_dis
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
#from models.multi_label import get_resnet_classifier
from torchvision.models import resnet18, resnet50
from models.multi_label import MultiLabelResNet

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")
parser.add_argument('--amp', action='store_true', help='use automatic half-precision')


def get_loss(output, target, index, cfg):
    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
    
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                pos_count = target.sum()
                neg_count = target.size()[0] - pos_count
                w0 = (neg_count + pos_count) / (2 * neg_count)
                w1 = (neg_count + pos_count) / (2 * pos_count)
                class_weights = torch.tensor([w0, w1], dtype=torch.float32)

                #weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1), target, pos_weight=class_weights[1])
                #print(f"output:{output}")
                #print(f"target:{target}")
        else:
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=pos_weight[index])

        label = torch.sigmoid(output[index].view(-1)).ge(0.5).float()
        acc = (target == label).float().sum() / len(label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return (loss, acc)

def deo_weighted_loss(p_vanilla, p_spurious, target, index, cfg, EPS, ce_loss):
    #weights = []
    #full_weights = torch.ones_like(target[:, index], dtype=torch.float32)  # Initialize weights for all samples as 1
    # Assuming output and output_spurious are logits from the model and bias discovery net respectively.     
    for target_val in [0, 1]:
        batch_bool = target[:, index].long() == target_val
        p_vanilla_label = p_vanilla[:, index]
        p_spurious_label = p_spurious[:, index]

        if target_val == 0:
            p_vanilla_label = 1 - p_vanilla_label

        positive_group_avg_p = (p_spurious_label[batch_bool] * p_vanilla_label[batch_bool]).sum() / (p_spurious_label[batch_bool].sum() + EPS)
        negative_group_avg_p = ((1 - p_spurious_label[batch_bool]) * p_vanilla_label[batch_bool]).sum() / ((1 - p_spurious_label[batch_bool]).sum() + EPS)

        # Adjust weights based on group averages
        if negative_group_avg_p < positive_group_avg_p:
            p_spurious_label[batch_bool] = 1 - p_spurious_label[batch_bool]


        # Ensure p_spurious_label[batch_bool] is one-dimensional to match full_weights[batch_bool]
        weight = p_spurious_label[batch_bool]

        weight += 1  # Squeeze to remove any singleton dimensions

        #weight = weight.unsqueeze(-1)
        # Apply the weight to the CE loss for the current label
        ce_loss[batch_bool, index] = ce_loss[batch_bool, index] * weight

        # Apply the weight to the CE loss for the current label
        #ce_loss[batch_bool, index] *= (p_spurious_label[batch_bool] + 1)       
        
    # Average the loss across all labels
    #return ce_loss.mean()
    return ce_loss


def calculate_deo_loss(p_vanilla, p_spurious, target, index, lambda_penalty, EPS):
    sum_discover_net_deo_loss = 0.0
    sum_penalty = 0.0
    for target_val in [0, 1]:
        batch_bool = target[:, index].long() == target_val
        p_vanilla_label = p_vanilla[:, index]
        p_spurious_label = p_spurious[:, index]

        if target_val == 0:
            p_vanilla_label = 1 - p_vanilla_label

        positive_group_avg_p = (p_spurious_label[batch_bool] * p_vanilla_label[batch_bool]).sum() / (p_spurious_label[batch_bool].sum() + EPS)
        negative_group_avg_p = ((1 - p_spurious_label[batch_bool]) * p_vanilla_label[batch_bool]).sum() / ((1 - p_spurious_label[batch_bool]).sum() + EPS)

        deo_loss = -torch.log(EPS + torch.abs(positive_group_avg_p - negative_group_avg_p))
        sum_discover_net_deo_loss += deo_loss.item()

        penalty = -torch.log(EPS + 1 - torch.abs(p_spurious_label.mean() - (1 - p_spurious_label).mean()))

        sum_penalty += penalty.item()
    sum_discover_net_deo_loss *= lambda_penalty
    sum_penalty *= lambda_penalty

    return torch.tensor(deo_loss, device=device, requires_grad=True), torch.tensor(penalty, device=device, requires_grad=True)


## With class weight:
    #def _setup_criterion(self):
       # w0 = (self.count_0 + self.count_1) / (2 * self.count_0)
      #  w1 = (self.count_0 + self.count_1) / (2 * self.count_1)
     #   class_weights = torch.tensor([w0, w1], dtype=torch.float32)

      #  if self.args.criterion == "BCE":
       #     self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1], reduction="none")
       # else:
      #      assert self.args.criterion == "CE"
      #      self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")

#def criterion(output, target, cfg, device):
  #  target = target.float()
   # if cfg.criterion == 'BCE':
  #      pos_weight = None
  #      if 'pos_weight' in cfg:
  #          pos_weight = torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device) 
  #      loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
  #      loss = loss_fn(output, target)
  #  else:
  #      raise ValueError(f"Unknown criterion: {cfg.criterion}")
    

   # return loss


def criterion(output, target, cfg, device):
    target = target.float()
    if cfg.criterion == 'BCE':
        for i in range(4):
            if cfg.get('batch_weight', False):
                target_ = target[:, i].view(-1)
                pos_weight = torch.from_numpy(
                np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target_)
                # Calculate weights for each class based on the batch's label distribution
                pos_counts = target_.sum(dim=0)
                neg_counts = target_.size(0) - pos_counts
                #print(f"pos_counts:{pos_counts}, neg_counts:{neg_counts}")
                # Avoid division by zero
                pos_counts = pos_counts.clamp(min=1)
                neg_counts = neg_counts.clamp(min=1)
                total_counts = pos_counts + neg_counts
                
                # Compute class weights
                w0 = (total_counts / (2.0 * neg_counts)).to(device)
                w1 = (total_counts / (2.0 * pos_counts)).to(device)
                class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                #class_weights = torch.where(target == 1, w1.expand_as(target), w0.expand_as(target))
                output_ = output[:, i].view(-1)
                #print(f"output2:{output.shape}, target2:{target.shape}, class_weights:{class_weights.shape}")
                # Compute weighted binary cross entropy loss
                loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=class_weights[1], reduction="none")
                #print(f"loss:{loss.shape}")
            else:
                pos_weight = torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device) 
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
                loss = loss_fn(output_, target_)

    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))


    return loss  # Note: Change reduction to "mean" to match the expected loss output

#def criterion(output, target, cfg, device):
    target = target.float()
    if cfg.criterion == 'BCE':
        pos_weight = None
        if 'pos_weight' in cfg:
            pos_weight = torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device) 
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        loss = loss_fn(output, target)
        print(f"loss:{loss.shape}")
    else:
        raise ValueError(f"Unknown criterion: {cfg.criterion}")
    

    return loss




def train_epoch(summary, summary_dev, cfg, args, model, bias_discover_net, dataloader,
                dataloader_dev, optimizer_bias_discover, optimizer_model, summary_writer, best_dict,
                dev_header):
    total_cls_loss = []
    total_ce_loss = []
    total_bias_discover_loss = []
    total_bias_discover_deo_loss = []
    total_bias_discover_penalty = []
    p_vanilla = None
    p_spurious = None

    torch.set_grad_enabled(True)
    device_ids = list(map(int, args.device_ids.split(',')))
    #device = torch.device('cuda:{}'.format(device_ids[0]))
    #print(f"device:{device}")
    second_train_loader = DataLoader(
        ImageDataset_Mayo(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    #scaler = GradScaler(enabled=args.amp)    
    steps = len(dataloader)
    #dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    num_tasks = len(cfg.num_classes)
    #print(f"num_tasks:{num_tasks}")

    accumulation_steps = 8  # Set accumulation steps
    accumulated_loss = 0.0  # For logging


    time_now = time.time()
    #loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
 
    for step, (main_data, second_data) in enumerate(zip(dataloader, second_train_loader)):
     # ============= start: train classifier net ================
        model.train()
        bias_discover_net.eval()
        image, target = main_data
        image = image.to(device)
        target = target.to(device)
        optimizer_model.zero_grad()
        optimizer_bias_discover.zero_grad()
        with autocast(enabled=args.amp):
            output, logit_map = model(image)
            #print(f"output:{output}")
            with torch.no_grad():
                output_spurious, logit_map_dis = bias_discover_net(image)

            
            for t in range(num_tasks):
                output_t = output[t]  # Get the output for task t
                label = torch.sigmoid(output_t.view(-1)).ge(0.5).float()
                target_t = target[:, t].view(-1)
                acc = (target_t == label).float().sum() / target_t.size(0)  # Fixed to use target_t.size(0) for clarity
                acc_sum[t] += acc.item()

            if isinstance(output, list):
                output = torch.cat(output, dim=1)
                #print(f"output:{output.shape}")
            if isinstance(output_spurious, list):
                output_spurious = torch.cat(output_spurious, dim=1)
                #print(f"output_spurious:{output_spurious.shape}")
            target = target.float()
            # Ensure label is in the correct shape [batch_size, num_classes]
            target = target.reshape(output.shape[0], -1)

            # Convert logits to probabilities
            #p_vanilla = torch.sigmoid(output)
            #p_spurious = torch.sigmoid(output_spurious)
            # different number of tasks
            loss = 0
            EPS = 1e-6  # To avoid division by zero


            ########
            #### deal with t
            with torch.cuda.amp.autocast(enabled=False):
                loss_sum_classifer = np.zeros(num_tasks)
                for t in range(num_tasks):
                    p_vanilla = torch.sigmoid(output)  # Adjust depending on actual output shape
                    #print(f"output:{output.shape}, output_spurious:{output_spurious.shape}")
                    p_spurious = torch.sigmoid(output_spurious) 
                    loss = criterion(output, target, cfg, device)
                    ce_loss = deo_weighted_loss(p_vanilla, p_spurious, target, t, cfg, EPS, loss)
                    
                    summed_ce_loss = ce_loss.sum(dim=0)
                    #print(f"summed_ce_loss:{summed_ce_loss}")
                    #summed_ce_loss_1_5 = summed_ce_loss.unsqueeze(0)
                    loss_sum_classifer[t] = summed_ce_loss[t].item() 
                # Accumulate this mean loss
                #print(f"loss_sum_classifer:{loss_sum_classifer}")
                ce_loss = summed_ce_loss.mean()/ accumulation_steps
                #ce_loss = loss_sum_classifer/ accumulation_steps
                #print(f"ce_loss:{ce_loss}")
                loss = ce_loss
                #loss_sum_classifer = ce_loss.sum()
        #optimizer.zero_grad()
        #print(f"loss1:{loss}")
        loss.backward()
        #optimizer.step()
        accumulated_loss += loss.item() * accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1 == steps):
            optimizer_model.step()  # Optimizer step for main model
            optimizer_bias_discover.step()  # Optimizer step for bias discover
            optimizer_model.zero_grad()
            optimizer_bias_discover.zero_grad()
        total_cls_loss = loss_sum_classifer
        #total_ce_loss = ce_loss.sum()
        # avg_cls_loss = total_cls_loss / (idx + 1)
        #avg_ce_loss = total_ce_loss / (step + 1)
        # ============= end: train classifier net ================

        # ============= start: train bias discover net ================
        bias_discover_net.train()
        model.eval()
        image, target = second_data
        image = image.to(device)
        target = target.to(device)
        optimizer_model.zero_grad()
        optimizer_bias_discover.zero_grad()
        with autocast(enabled=args.amp):
            with torch.no_grad():
                output, logit_map = model(image)
            output_spurious, logit_map_dis = bias_discover_net(image)

            if isinstance(output, list):
                output = torch.cat(output, dim=1)
                #print(f"output:{output.shape}")
            if isinstance(output_spurious, list):
                output_spurious = torch.cat(output_spurious, dim=1)
                #print(f"output_spurious:{output_spurious.shape}")
            target = target.float()
            # Ensure label is in the correct shape [batch_size, num_classes]
            target = target.reshape(output.shape[0], -1)


            # Convert logits to probabilities
            with torch.cuda.amp.autocast(enabled=False):
                total_loss = torch.tensor(0., device=device)
                lambda_penalty = 1
                EPS = 1e-6  # To avoid division by zero
                loss_sum_discover = np.zeros(num_tasks)
                lambda_penalty = 1
                for t in range(num_tasks):
                    p_vanilla = torch.sigmoid(output)  # Adjust depending on actual output shape
                    p_spurious = torch.sigmoid(output_spurious) 
                    deo_loss_t, penalty_t = calculate_deo_loss(p_vanilla, p_spurious, target, t, lambda_penalty, EPS)
                    # Directly add the losses to total_loss
                    total_loss += deo_loss_t + penalty_t * lambda_penalty
                    # Update the numpy arrays with the calculated losses
                    loss_sum_discover[t] += deo_loss_t.item()

                    # Normalize total_loss by accumulation steps if necessary
                total_loss = total_loss / accumulation_steps

            #loss_sum_discover = total_loss.sum()
            #total_bias_discover_loss += total_loss.item()
            #print(f"total_loss:{total_loss}")    
            #loss_sum[t] += total_loss.item()
        
        #scaler.scale(total_loss).backward()
        #print(f"loss2:{total_loss}")                
        total_loss.backward()
        accumulated_loss += loss.item() * accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1 == steps):
            optimizer_model.step()  # Optimizer step for main model
            optimizer_bias_discover.step()  # Optimizer step for bias discover
            optimizer_model.zero_grad()
            optimizer_bias_discover.zero_grad()
            
        #total_bias_discover_deo_loss = sum_discover_net_deo_loss.sum()
        #total_bias_discover_penalty = sum_penalty.sum()
        #loss_sum_discover = total_loss.sum()
        #avg_discover_net_deo_loss = total_bias_discover_deo_loss / (step + 1)
        #avg_discover_net_penalty = total_bias_discover_penalty / (step + 1)
        #avg_bias_discover_loss = loss_sum_discover / (step + 1)


        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            #print(f"cfg.log_every:{cfg.log_every}")
            #loss_sum /= cfg.log_every
            acc_sum = [acc / cfg.log_every for acc in acc_sum]
            #acc_sum /= cfg.log_every
            loss_sum_classifer/= cfg.log_every
            loss_sum_discover/= cfg.log_every


            loss_str_cls = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum_classifer))
            loss_str_dis = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum_discover))
            acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))



            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss1 : {}, Loss2 : {}'
                'Acc : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_str_cls, loss_str_dis,
                        acc_str, time_spent))

            for t in range(num_tasks):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(label_header[t]), loss_sum_classifer[t], loss_sum_discover[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/acc_{}'.format(label_header[t]), acc_sum[t],
                    summary['step'])

            loss_sum_classifer = np.zeros(num_tasks)
            loss_sum_discover = np.zeros(num_tasks)
            acc_sum = np.zeros(num_tasks)

        if summary['step'] % cfg.test_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
            time_spent = time.time() - time_now

            auclist = []
            for i in range(len(cfg.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auclist.append(auc)
            summary_dev['auc'] = np.array(auclist)

            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['loss']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['acc']))
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))

            logging.info(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
                'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))

            for t in range(len(cfg.num_classes)):
                summary_writer.add_scalar(
                    'dev/loss_{}'.format(dev_header[t]),
                    summary_dev['loss'][t], summary['step'])
                summary_writer.add_scalar(
                    'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                    summary['step'])

            save_best = False
            mean_acc = summary_dev['acc'][cfg.save_index].mean()
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_auc = summary_dev['auc'][cfg.save_index].mean()
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc':
                    save_best = True

            mean_loss = summary_dev['loss'][cfg.save_index].mean()
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

            if save_best:
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'state_dict': model.module.state_dict()},
                    os.path.join(args.save_path, 'best{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                logging.info(
                    '{}, Best, Step : {}, Loss : {}, Acc : {},Auc :{},'
                    'Best Auc : {:.3f}' .format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        auc_dev_str,
                        best_dict['auc_dev_best']))
        model.train()
        torch.set_grad_enabled(True)
    summary['epoch'] += 1

    return summary, best_dict

#######

def test_epoch(summary, cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    #device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)

    predlist = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))
    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output, logit_map = model(image)
        # different number of tasks
        for t in range(len(cfg.num_classes)):

            loss_t, acc_t = get_loss(output, target, t, cfg)
            #loss_t = criterion(output, target, t, cfg, device)
            # AUC
            output_tensor = torch.sigmoid(
                output[t].view(-1)).cpu().detach().numpy()
            target_tensor = target[:, t].view(-1).cpu().detach().numpy()
            if step == 0:
                predlist[t] = output_tensor
                true_list[t] = target_tensor
            else:
                predlist[t] = np.append(predlist[t], output_tensor)
                true_list[t] = np.append(true_list[t], target_tensor)

            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary, predlist, true_list

def get_resnet_classifier(resnet_type='resnet18', num_classes=None, pretrained=False):
    # Ensure num_classes is provided and is a list
    if num_classes is None:
        raise ValueError("num_classes must be provided and must be a list of integers.")
    
    original_resnet = resnet18(pretrained=pretrained) if resnet_type == 'resnet18' else resnet50(pretrained=pretrained)
    model = MultiLabelResNet(original_resnet, num_classes)
    return model

def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    #if num_devices < len(device_ids):
     #   raise Exception(
     #       '#available gpu : {} < --device_ids : {}'
     #       .format(num_devices, len(device_ids)))
    #device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    #bias_discover_net = get_resnet_classifier(resnet_type='resnet50', num_classes=cfg.num_classes, pretrained=False)
    bias_discover_net = Classifier_dis(cfg)
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))
    #model = DataParallel(model, device_ids=device_ids).to(device).train()
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    bias_discover_net = DataParallel(bias_discover_net, device_ids=device_ids).to(device).train()
    if args.pre_train is not None:
        if os.path.exists(args.pre_train):
            ckpt = torch.load(args.pre_train, map_location=device)
            #model.module.load_state_dict(ckpt)
            model.module.load_state_dict(ckpt, strict=False)
            model.classifier = torch.nn.Identity()
            model.fc_0 = torch.nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1))
            model.fc_1 = torch.nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1))
            model.fc_2 = torch.nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1))
            model.fc_3 = torch.nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1))
            model.bn_0 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model.bn_1 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model.bn_2 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model.bn_3 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    optimizer_model = get_optimizer(model.parameters(), cfg)
    optimizer_bias_discover = torch.optim.Adam(bias_discover_net.parameters(), lr=1e-4)  # Replace with actual learning rate


    src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
    dst_folder = os.path.join(args.save_path, 'classification')
    rc, size = subprocess.getstatusoutput('du --max-depth=0 %s | cut -f1'
                                          % src_folder)
    if rc != 0:
        raise Exception('Copy folder error : {}'.format(rc))
    rc, err_msg = subprocess.getstatusoutput('cp -R %s %s' % (src_folder,
                                                              dst_folder))
    if rc != 0:
        raise Exception('copy folder error : {}'.format(err_msg))

    copyfile(cfg.train_csv, os.path.join(args.save_path, 'train.csv'))
    copyfile(cfg.dev_csv, os.path.join(args.save_path, 'dev.csv'))

    dataloader_train = DataLoader(
        ImageDataset_Mayo(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    print(f"dataloader_train:{len(dataloader_train)}")
    dataloader_dev = DataLoader(
        ImageDataset_Mayo(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    dev_header = dataloader_dev.dataset._label_header

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    #tensorbd_dir = "/mnt/storage/jialu/Chexpert-master/runs/test" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Check if the directory exists, and create it if it doesn't
   # if not os.path.exists(tensorbd_dir):
     #   os.makedirs(tensorbd_dir)  # Use os.makedirs() to create intermediate directories if necessary
    #else:
    #    print(f"Directory {tensorbd_dir} already exists.")

    # Proceed with using the directory, e.g., for TensorBoard logs
    #summary_writer = SummaryWriter(tensorbd_dir)
    summary_writer = SummaryWriter(args.save_path)
    #print("summarywriter")
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1}

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        #ckpt_path = args.resume
        #ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        epoch_start = ckpt['epoch']
    #print("i'm here")
    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer_model.param_groups:
            param_group['lr'] = lr

        for param_group in optimizer_bias_discover.param_groups:
            param_group['lr'] = lr


        summary_train, best_dict = train_epoch(summary_train, summary_dev, cfg, 
        args, model, bias_discover_net, dataloader_train, dataloader_dev, 
        optimizer_bias_discover, optimizer_model,
        summary_writer, best_dict, dev_header
)

        time_now = time.time()
        summary_dev, predlist, true_list = test_epoch(
            summary_dev, cfg, args, model, dataloader_dev)
        time_spent = time.time() - time_now

        auclist = []
        for i in range(len(cfg.num_classes)):
            y_pred = predlist[i]
            y_true = true_list[i]
            fpr, tpr, thresholds = metrics.roc_curve(
                y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auclist.append(auc)
        summary_dev['auc'] = np.array(auclist)

        loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                    summary_dev['loss']))
        acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['acc']))
        auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['auc']))

        logging.info(
            '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
            'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                acc_dev_str,
                auc_dev_str,
                summary_dev['auc'].mean(),
                time_spent))

        for t in range(len(cfg.num_classes)):
            summary_writer.add_scalar(
                'dev/loss_{}'.format(dev_header[t]), summary_dev['loss'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                summary_train['step'])

        save_best = False

        mean_acc = summary_dev['acc'][cfg.save_index].mean()
        if mean_acc >= best_dict['acc_dev_best']:
            best_dict['acc_dev_best'] = mean_acc
            if cfg.best_target == 'acc':
                save_best = True

        mean_auc = summary_dev['auc'][cfg.save_index].mean()
        if mean_auc >= best_dict['auc_dev_best']:
            best_dict['auc_dev_best'] = mean_auc
            if cfg.best_target == 'auc':
                save_best = True

        mean_loss = summary_dev['loss'][cfg.save_index].mean()
        if mean_loss <= best_dict['loss_dev_best']:
            best_dict['loss_dev_best'] = mean_loss
            if cfg.best_target == 'loss':
                save_best = True

        if save_best:
            torch.save(
                {'epoch': summary_train['epoch'],
                 'step': summary_train['step'],
                 'acc_dev_best': best_dict['acc_dev_best'],
                 'auc_dev_best': best_dict['auc_dev_best'],
                 'loss_dev_best': best_dict['loss_dev_best'],
                 'state_dict': model.module.state_dict()},
                os.path.join(args.save_path,
                             'best{}.ckpt'.format(best_dict['best_idx']))
            )
            best_dict['best_idx'] += 1
            if best_dict['best_idx'] > cfg.save_top_k:
                best_dict['best_idx'] = 1
            logging.info(
                '{}, Best, Step : {}, Loss : {}, Acc : {},'
                'Auc :{},Best Auc : {:.3f}' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    best_dict['auc_dev_best']))
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'acc_dev_best': best_dict['acc_dev_best'],
                    'auc_dev_best': best_dict['auc_dev_best'],
                    'loss_dev_best': best_dict['loss_dev_best'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))
    summary_writer.close()


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()
