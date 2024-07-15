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

from data.dataset_Mayo_test import ImageDataset_Mayo_test # noqa
from model.classifier_conf import ConfClassifier  # noqa

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--in_csv_path', default='dev.csv', metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--out_csv_path', default='test/causalconf_viewhealth_allinvert.csv',
                    metavar='OUT_CSV_PATH', type=str,
                    help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")

if not os.path.exists('test'):
    os.mkdir('test')


def get_pred(output, cfg):
    if cfg.criterion_target == 'BCE' or cfg.criterion_target == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion_target))
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

    task_names = ['MACE']


    bias_names = [
        'AGE_AT_XRAY',
        'Gender',
        'Calcium_Group'
    ]

    test_header = ['img_path'] + ['pred_' + task for task in task_names]



    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')

      
        for step in range(steps):
            image, path, labels = next(dataiter)
            image = image.to(device)
            causal_output, output, conf_output = model(image)
            batch_size = len(path)
            pred = np.zeros((num_tasks, batch_size))
            label = np.zeros((num_tasks, batch_size))



            for i in range(num_tasks):
                pred[i] = get_pred(output[i], cfg)
                
                
            for i in range(batch_size):
               # if isinstance(bias[i], torch.Tensor):
                    # If bias is a tensor, ensure it's moved to CPU and converted to numpy
                #    bias_attributes = bias[i].cpu().numpy()
               # else:
                    # Assuming bias[i] is already a list or a numpy array
              #      bias_attributes = bias[i]
                # Convert all bias attributes to strings for CSV writing
               # bias_str = ','.join(map(str, bias_attributes))
                batch = ','.join(map(lambda x: '{}'.format(x), pred[:, i]))
                #result = path[i] + ',' + bias_str + ',' + batch + ',' + binary_preds_str + ',' + label[i]
                result = path[i] + ',' + batch 
                f.write(result + '\n')
                logging.info('{}, Image : {},  Prob : {} '.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), path[i], batch))
    






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
        ImageDataset_Mayo_test(args.in_csv_path, cfg, mode='test'),
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
