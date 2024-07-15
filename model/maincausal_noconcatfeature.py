from torch import nn
import torch

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap

#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class CausalClassifier(nn.Module):

    def __init__(self, cfg):
        super(CausalClassifier, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg)
        self.global_pool = GlobalPool(cfg)
        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3
        self._init_classifier_main()
        self._init_bn_main()
        self._init_attention_map()
        self._init_classifier_causal()
        self._init_bn_causal()

        
    def _init_classifier_main(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        self.backbone.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier_main = getattr(self, "fc_" + str(index))
            if isinstance(classifier_main, nn.Conv2d):
                classifier_main.weight.data.normal_(0, 0.01)
                classifier_main.bias.data.zero_()
    
    
    def _init_classifier_causal(self):
        for index, num_class in enumerate(self.cfg.num_causal):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_causal_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_causal_" +
                    str(index),
                    nn.Conv2d(
                        self.backbone.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(
                    self,
                    "fc_causal_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier_causal = getattr(self, "fc_causal_" + str(index))
            if isinstance(classifier_causal, nn.Conv2d):
                classifier_causal.weight.data.normal_(0, 0.01)
                classifier_causal.bias.data.zero_()


    def _init_bn_main(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )
            

    def _init_bn_causal(self):
        for index, num_class in enumerate(self.cfg.num_causal):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "bn_causal_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_causal_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "bn_causal_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

    def _init_attention_map(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            setattr(
                self,
                "attention_map",
                AttentionMap(
                    self.cfg,
                    self.backbone.num_features))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
    
    
    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.backbone(x)
        # [(N, 1), (N,1),...]
        causal_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_causal]
        main_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_classes]
        #causal_logits = [None] * len(self.cfg.num_causal)
        #main_logits = [None] * len(self.cfg.num_classes)

        #causal_logits = list()
        #main_logits = list()
        # [(N, H, W), (N, H, W),...]
        causal_logit_maps = list()
        main_logit_maps = list()
        pooled_features_list = []  # To store pooled features from each iteration
    
        for index, num_class in enumerate(self.cfg.num_causal):     
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier_causal = getattr(self, "fc_causal_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            if not (self.cfg.global_pool == 'AVG_MAX' or
                    self.cfg.global_pool == 'AVG_MAX_LSE'):
                logit_map = classifier_causal(feat_map)
                causal_logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)

            if self.cfg.fc_bn:
                bn_causal = getattr(self, "bn_causal_" + str(index))
                feat = bn_causal(feat)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
           # print(f"feat:{feat.shape}")
            # (N, num_class, 1, 1)

            logit = classifier_causal(feat)
            #print(f"logit:{logit.shape}")
            # (N, num_class)
            

            logit = logit.squeeze(-1).squeeze(-1)
            #print(f"logit_causal:{logit.shape}")

            causal_logits[index] = logit
            #causal_logits.append(logit)


        
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier_main = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            if not (self.cfg.global_pool == 'AVG_MAX' or
                    self.cfg.global_pool == 'AVG_MAX_LSE'):
                logit_map = classifier_main(feat_map)
                main_logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            pooled_feat = self.global_pool(feat_map, logit_map)
            

            if self.cfg.fc_bn:
                bn_layer = getattr(self, "bn_" + str(index))
                combined_features = bn_layer(pooled_feat)

            #if self.cfg.fc_bn:
            #    bn_main = getattr(self, "bn_" + str(index))
                #print(f"combined_features:{combined_features.shape}")
             #   combined_features = bn_main(combined_features)
            combined_features = F.dropout(combined_features, p=self.cfg.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier_main(combined_features)
            
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            #print(f"logit_main:{logit.shape}")
            
            main_logits[index] = logit
            


        return (causal_logits, causal_logit_maps, main_logits, main_logit_maps)

        