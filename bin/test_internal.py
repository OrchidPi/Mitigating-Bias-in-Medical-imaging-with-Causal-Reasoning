import os
import sys
import argparse
import logging
import json
import time
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
#from ml_inference import SplineCalib
#from sklearn.isotonic import IsotonicRegression
#from sklearn.linear_model import LogisticRegression
#from temeperature_scaling import ModelWithTemperature
import pandas as pd



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Primary device

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset_Mayo import ImageDataset_Mayo # noqa
from model.classifier_conf import ConfClassifier  # noqa

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--in_csv_path', default='dev.csv', metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--out_csv_path', default='test/causalconf_view_allinvert.csv',
                    metavar='OUT_CSV_PATH', type=str,
                    help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")

if not os.path.exists('test'):
    os.mkdir('test')


def get_pred(output, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return pred

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def test_epoch(cfg, args, model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    #device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)
    pred_lists = [[] for _ in range(num_tasks)]
    true_lists = [[] for _ in range(num_tasks)]
    binary_pred_lists = [[] for _ in range(num_tasks)]


    task_names = [
        'MACE_6M',
        'MACE_1yr',
        'MACE_2yr',
        'MACE_5yr']


    

    test_header = ['img_path'] + ['pred_' + task for task in task_names] 
    
    bias_names = [
        'AGE_AT_XRAY',
        'Gender',
        'Chf',
        'Diabetes',
        'Hypertension',
        'CKD'
    ]

    causal_names = [
        'Chf',
        'CKD'
    ]

    test_header = ['img_path'] +  bias_names + ['pred_' + task for task in task_names]




    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')
        total_auc = [0 for _ in range(num_tasks)]
        total_accuracy = [0 for _ in range(num_tasks)]
        total_zeros_per_task = [0] * num_tasks
        total_ones_per_task = [0] * num_tasks

       # y1 = np.zeros([steps, batch_size,num_tasks])
       # y2 = np.zeros([steps, batch_size,num_tasks])
        for step in range(steps):
            image, path, bias, labels = next(dataiter)
            image = image.to(device)
            causal_output, output, conf_output = model(image)
            batch_size = len(path)
            pred = np.zeros((num_tasks, batch_size))
            label = np.zeros((num_tasks, batch_size))



            for i in range(num_tasks):
                pred[i] = get_pred(output[i], cfg)
                #pred_lists[i].extend(pred[i])

                label[i] = labels[:, i].view(-1).cpu().detach().numpy()
                #true_lists[i].extend(label[i])
                
               # y1[step, :, i] = pred[i]
              #  y2[step, :, i] = label[i]
        
                #y_pred = np.array(pred_lists[i])
                #y_true = np.array(true_lists[i])


                #optimal_threshold = Find_Optimal_Cutoff(y_true, y_pred)[0]
                #optimal_threshold = np.array([optimal_threshold])

                #binary_preds = (y_pred >= optimal_threshold).astype(int)
                #binary_pred_lists[i].extend(binary_preds.tolist())         

            for i in range(batch_size):
                #binary_preds_str = ','.join([str(binary_pred_lists[task][i]) for task in range(num_tasks)])  # Format binary predictions as a string
                #true_labels_str = ','.join([str(true_lists[task][i]) for task in range(num_tasks)])

                if isinstance(bias[i], torch.Tensor):
                    # If bias is a tensor, ensure it's moved to CPU and converted to numpy
                    bias_attributes = bias[i].cpu().numpy()
                else:
                    # Assuming bias[i] is already a list or a numpy array
                    bias_attributes = bias[i]
                # Convert all bias attributes to strings for CSV writing
                bias_str = ','.join(map(str, bias_attributes))
                #true_str = ','.join(map(str, label[i]))
                batch = ','.join(map(lambda x: '{}'.format(x), pred[:, i]))
                #result = path[i] + ',' + bias_str + ',' + batch + ',' + binary_preds_str + ',' + label[i]
                result = path[i] + ',' + bias_str + ',' + batch
                f.write(result + '\n')
                logging.info('{}, Image : {}, Bias : {}, Prob : {} '.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), path[i], bias_str, batch))
        

      

                #binary_accuracy = np.mean(binary_preds == y_true)


                #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
              #  auc = metrics.auc(fpr, tpr)


               # total_auc[i] += auc  # Add the AUC for this task to the total sum
               # total_accuracy[i] += binary_accuracy  # Add the accuracy for this task to the total sum


               # print(f"Task {i}: Optimal Binary Accuracy = {binary_accuracy:.3f}")
               # print(f"Task {i}: AUC = {auc:.3f}")
    

        average_auc = 0
        average_accuracy = 0
        for i in range(num_tasks):
            average_auc = total_auc[i]/steps
            average_accuracy = total_accuracy[i]/steps

           # print(f"Task {i}: Accuracy = {average_accuracy:.3f}")
           # print(f"Task {i}: AUC = {average_auc:.3f}")        
        
                






def run(args):
    with open(args.model_path + 'cfg.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    #if num_devices < len(device_ids):
     #   raise Exception(
     #       '#available gpu : {} < --device_ids : {}'
    #        .format(num_devices, len(device_ids)))
    #device = torch.device('cuda:{}'.format(device_ids[0]))

    model = ConfClassifier(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt_path = os.path.join(args.model_path, 'best1.ckpt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.module.load_state_dict(ckpt['state_dict'], strict=False)

    dataloader_test = DataLoader(
        ImageDataset_Mayo(args.in_csv_path, cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    
    dataloader_dev = DataLoader(
        ImageDataset_Mayo(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)


    test_epoch(cfg, args, model, dataloader_test, args.out_csv_path)

    print('Save best is step :', ckpt['step'], 'AUC :', ckpt['auc_dev_best'])


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
