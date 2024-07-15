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
#from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Primary device

torch.manual_seed(2)
torch.cuda.manual_seed_all(50)

#from data.datasetCXP import ImageDataset_CXP
#from data.datasetNIH import ImageDataset_NIH
from data.dataset_Mayo import ImageDataset_Mayo
from model.mainconf import ConfmainClassifier
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
#from model.loss_models import MultiNoiseLoss


CUDA_LAUNCH_BLOCKING="1"

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


def get_loss_main(output, target, index, cfg):
    if cfg.criterion_target == 'BCE':
        #print(f"target1:{target.shape}")
        #print(f"output_len:{len(output)}, output[0]:{output[0].shape}")
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1).to(device) 
        
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1).to(device), target, pos_weight=weight)
                #print(f"output:{output}")
                #print(f"target:{target}")
        else:
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1).to(device), target, pos_weight=pos_weight[index])

        label = torch.sigmoid(output[index].view(-1).to(device)).ge(0.5).float()
        acc = (target == label).float().sum() / len(label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion_target))

    return (loss, acc)


def get_loss_conf_gender(output, target):
    #print(f"target1:{target.shape}")
    #print(f"output1:{output.shape}")
    target = target.view(-1).to(device)
    loss = F.binary_cross_entropy_with_logits(
        output.view(-1).to(device), target)

    label = torch.sigmoid(output.view(-1).to(device)).ge(0.5).float()
    acc = (target == label).float().sum() / len(label)  

    return (loss, acc)

def get_loss_conf_age(output, target):
    #print(f"target2:{target.shape}")
    #print(f"output2:{output.shape}")
    target = target.long().to(device) 
    target_ = target
    target_ = torch.nn.functional.one_hot(target_, num_classes=output.shape[1])
    target_ = target_.float()
    #print(f"target2_new:{target_.shape}")

    loss = F.cross_entropy(output.to(device), target_)

    _, preds = torch.max(output.to(device), dim=1)
    acc = (preds == target).float().mean()

    return (loss, acc)




def confusion_loss(pred_pros):
    alpha = -10
    a, b = pred_pros.shape
    uniform_probs = torch.FloatTensor(a,b).uniform_(0,1).to(device)
    return alpha * (torch.sum(uniform_probs*torch.log(pred_pros))/float(pred_pros.size(0)))


def train_epoch(summary, summary_dev, cfg, args, model, dataloader,
                dataloader_dev, optimizer, summary_writer, best_dict,
                dev_header, dev_confounder_header):
    torch.set_grad_enabled(True)
    model.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    #device = torch.device('cuda:{}'.format(device_ids[0]))
    #print(f"device:{device}")
    steps = len(dataloader)
    dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    bias_header = dataloader.dataset._bias_header
    num_tasks = len(cfg.num_classes)
    num_conf = len(cfg.num_conf)
    #print(f"num_tasks:{num_tasks}")

    accumulation_steps = 8  # Set accumulation steps
    optimizer.zero_grad()  # Reset gradients accumulation
    accumulated_loss = 0.0  # For logging
    accumulated_loss_conf = 0.0


    time_now = time.time()
    mainloss_sum = np.zeros(num_tasks)
    confloss_sum = np.zeros(num_conf)
    mainacc_sum = np.zeros(num_tasks)
    confacc_sum = np.zeros(num_conf)


    #total_zeros_per_task = [0] * num_tasks
   # total_ones_per_task = [0] * num_tasks
    for step in range(steps):
        image, target, bias, causal_attr = next(dataiter)
        image = image.to(device)
        target = target.to(device)

        output, conf_output = model(image)
    

        
        confloss = 0
        mainloss = 0
        total_loss = 0
        uniform_loss = 0
        losses = []
        losses_uniform = []
   

        
        for j in range(num_conf):
            if j == 0:  # Assuming index 0 is for age
                loss_j, acc_j = get_loss_conf_age(conf_output[j], bias[:, j])
            elif j == 1:  # Assuming index 1 is for gender
                loss_j, acc_j = get_loss_conf_gender(conf_output[j], bias[:, j])
            
            confloss += loss_j / accumulation_steps
            confloss_sum[j] += loss_j.item()
            confacc_sum[j] += acc_j.item()

            conf_probs = F.softmax(conf_output[j], dim=1)

            conf_uniform_loss = confusion_loss(conf_probs)
            

            uniform_loss += conf_uniform_loss / accumulation_steps

            losses_uniform.append(conf_uniform_loss)
            losses.append(loss_j)





        for t in range(num_tasks):
            loss_t, acc_t = get_loss_main(output, target, t, cfg)
            #print(f"ce_loss.shape: {loss_t.shape}")
            #loss += loss_t
            mainloss += loss_t /accumulation_steps
            mainloss_sum[t] += loss_t.item()
            mainacc_sum[t] += acc_t.item()

 
        total_loss = mainloss 

       # optimizer.zero_grad()
        total_loss.backward()
       # optimizer.step()

        #optimizer.step()
        accumulated_loss += total_loss.item() * accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1 == steps):
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients


        total_confounderloss = confloss + uniform_loss
        total_confounderloss.backward()
        accumulated_loss_conf += total_confounderloss.item() * accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1 == steps):
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients


        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            mainloss_sum /= cfg.log_every
            mainacc_sum /= cfg.log_every
            #uniform_loss /= cfg.log_every
            confacc_sum /= cfg.log_every
            if total_loss.dim() == 0:  # Check if it's a scalar tensor
                loss_str = '{:.5f}'.format(total_loss.item())  # Directly format the scalar value
            else:
                loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), total_loss))  # Iterate over elements

            #loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), total_loss))
            loss_main = ' '.join(map(lambda x: '{:.5f}'.format(x), mainloss_sum))
            loss_conf = '{:.5f}'.format(uniform_loss.item())
            acc_main = ' '.join(map(lambda x: '{:.3f}'.format(x), mainacc_sum))
            acc_conf = ' '.join(map(lambda x: '{:.3f}'.format(x), confacc_sum))



            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {},  Mainloss:{}, Confounderloss:{},'
                'MainAcc : {}, Confounderacc:{}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_str, loss_main, loss_conf,
                        acc_main, acc_conf, time_spent))

            for t in range(num_tasks):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(label_header[t]), mainloss_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/acc_{}'.format(label_header[t]), mainacc_sum[t],
                    summary['step'])

            mainloss_sum = np.zeros(num_tasks)
            mainacc_sum = np.zeros(num_tasks)

            for j in range(num_conf):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(bias_header[j]), confloss_sum[j],
                    summary['step'])
                
            confloss_sum = np.zeros(num_conf)
            confacc_sum = np.zeros(num_conf)
            

        if summary['step'] % cfg.test_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
            time_spent = time.time() - time_now

            auclist_main = []


            for i in range(len(cfg.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auclist_main.append(auc)
            summary_dev['auc'] = np.array(auclist_main)




            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['loss']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['acc']))
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))
          

            logging.info(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {}, '
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
    #for task in range(num_tasks):
     #   print(f"Task {task + 1} - Total number of zeros: {total_zeros_per_task[task]}, Total number of ones: {total_ones_per_task[task]}")

    return summary, best_dict


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
        image, target, bias, causal_attr = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output, conf_output = model(image)
        # different number of tasks
        for t in range(len(cfg.num_classes)):

            #loss_t, acc_t = get_loss(output, target, t, device, cfg)
            loss_t, acc_t = get_loss_main(output, target, t, cfg)
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
    #num_devices = torch.cuda.device_count()
    #if num_devices < len(device_ids):
    #    raise Exception(
    #        '#available gpu : {} < --device_ids : {}'
     #       .format(num_devices, len(device_ids)))
    #device = torch.device('cuda:{}'.format(device_ids[0]))

    model = ConfmainClassifier(cfg)
    #model = ConfClassifier(cfg).to(device)
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))
    #model = DataParallel(model, device_ids=device_ids).to(device).train()
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    if args.pre_train is not None:
        if os.path.exists(args.pre_train):
            ckpt = torch.load(args.pre_train, map_location=device)
            filtered_state_dict = {k: v for k, v in ckpt.items() if k in model.state_dict() and v.size() == model.state_dict()[k].size()}
            model.load_state_dict(filtered_state_dict, strict=False)
            

    optimizer = get_optimizer(model.parameters(), cfg)

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

    dataloader_dev = DataLoader(
        ImageDataset_Mayo(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    dev_header = dataloader_dev.dataset._label_header
    dev_confounder_header = dataloader_dev.dataset._bias_header


    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1}

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        #ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'], strict=False)
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



        summary_train, best_dict = train_epoch(
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header, dev_confounder_header)

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
