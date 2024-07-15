# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation



import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

CUDA_LAUNCH_BLOCKING=1

def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
        kernel_count = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Linear(kernel_count, 3), nn.Softmax())
        net = net.float()
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # 加载指定路径的权重参数
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        ckpt = torch.load(weight_path)
        filtered_state_dict = {k: v for k, v in ckpt.items() if k in net.state_dict() and v.size() == net.state_dict()[k].size()}
        net.load_state_dict(filtered_state_dict, strict=False)        
        #net.load_state_dict(state_dict)
    elif weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


#def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)



def main(args):
    # 输入
    #img = io.imread(args.image_path)
    image_directory = args.image_path
    images = [img for img in os.listdir(image_directory) if img.endswith((".png"))]

    for image_name in images:
        image_path = os.path.join(image_directory, image_name)
        img = cv2.imread(image_path, 0)
        img = 255 - img
        # Save the inverted image
        img = Image.fromarray(img)
        img = GetTransforms(img, type='None')
        img = np.array(img)
        img = transform(img)

        inverted_image_path = os.path.join(args.output_dir, f"inverted_{image_name}")
        cv2.imwrite(inverted_image_path, img)
        
        img = cv2.imread(inverted_image_path)


       
        # Paths to your images
        #reference_image_path = './result_new/orignal_80-_chf2.png'  # Adjusted to your provided path
        #reference_image_path = 'your_reference_image_path_here'  # You need to specify this

        # Reading the images. Assuming the reference image is also to be read in grayscale
        #source_img = img
        #print(f"image0:{source_img.shape}")
        #reference_img = cv2.imread(reference_image_path)  # Change 0 to 1 if the reference image is in color
        #print(f"image0:{reference_img.shape}")


        # Perform histogram matching. Note: If the reference image is in color, you need to adjust this part
        # as match_histograms requires both images to have the same number of channels
        #matched_img = exposure.match_histograms(source_img, reference_img, channel_axis=-1)

        # Convert matched_img back to a format that can be saved by OpenCV, if necessary
        #matched_img_cv2 = (matched_img * 255).astype('uint8')  # Normalize back to 0-255 for OpenCV
        #matched_img_cv2= 255 - matched_img_cv2    

        # Save the matched image
        #matched_image_path = os.path.join(args.output_dir, f"matched_image_{image_name}")
        #cv2.imwrite(matched_image_path, matched_img_cv2)

        
        #img = cv2.imread(args.image_path, 0)
        #img = cv2.imread(matched_image_path, 0)
        img = cv2.imread(inverted_image_path,0)
        #print(f"image0:{img.shape}")
        img = np.float32(cv2.resize(img, (224,224))) / 255
        #print(f"image1:{img.shape}")
        image = np.zeros([img.shape[0], img.shape[1],3])
        #print(f"image2:{image.shape}")
        image[:,:,0] = img
        image[:,:,1] = img
        image[:,:,2] = img
        inputs = prepare_input(image)
        # 输出图像
        image_dict = {}
        # 网络
        net = get_net(args.network, args.weight_path)
        # Grad-CAM
        layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
        grad_cam = GradCAM(net, layer_name)
        mask = grad_cam(inputs, args.class_id)  # cam mask
        image_dict['cam'], image_dict['heatmap'] = gen_cam(image, mask)
        grad_cam.remove_handlers()
        # Grad-CAM++
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
        image_dict['cam++'], image_dict['heatmap++'] = gen_cam(image, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()

        # GuidedBackPropagation
        gbp = GuidedBackPropagation(net)
        inputs.grad.zero_()  # 梯度置零
        grad = gbp(inputs)

        gb = gen_gb(grad)
        image_dict['gb'] = norm_image(gb)
        # 生成Guided Grad-CAM
        cam_gb = gb * mask[..., np.newaxis]
        image_dict['cam_gb'] = norm_image(cam_gb)

        #save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)
        save_image(image_dict, os.path.basename(image_path), args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./others',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='./others_result',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
