from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch


class ResNet(nn.Module):
    def __init__(self, args, net, pretrained=True):
        super(ResNet, self).__init__()

        # backbone
        if net == 'resnet50':
            model_resnet = models.resnet50(pretrained=pretrained)
            inc = 2048
        else:
            raise NotImplementedError

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        # aada-psi classifier
        self.classifier = PSIClassifier(num_class=args.ncls, inc=inc, num_emb=512, temp=0.1)

        self.frozen_layer_list = []

    def forward(self, x, getemb=False, getfeat=False, justclf=False):
        assert (not (getemb and getfeat))

        if not justclf:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
        else:
            x = x

        feat = x.view(x.size(0), -1)
        out = self.classifier(feat, getemb=getemb)
        out = (out, feat) if getfeat else out

        return out

    def trainable_parameters(self):
        backbone_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        backbone_params = []
        for layer in backbone_layers:
            backbone_params += [param for param in layer.parameters()]
        classifier_params = list(self.classifier.parameters())

        return backbone_params, classifier_params

    def freeze_layers(self, layer_list):
        for layer in layer_list:
            # no gradient
            for param in layer.parameters():
                param.requires_grad = False

            # eval mode for batchnorm
            for layer in layer_list:
                for m in layer.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        self.frozen_layer_list = layer_list

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        # eval mode on frozen layers
        for layer in self.frozen_layer_list:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        return self

    def set_bn_mode(self, mode='train'):
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                if mode == 'train':
                    m.train()
                elif mode == 'eval':
                    m.eval()
                else:
                    raise NotImplementedError

    def set_bn_requires_grad(self, requires_grad=True):
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = requires_grad
                m.bias.requires_grad = requires_grad


class PSIClassifier(nn.Module):
    def __init__(self, inc=4096, num_emb=512, num_class=64, temp=0.05):
        super(PSIClassifier, self).__init__()
        self.fc1 = nn.Linear(inc, num_emb)
        self.bn = nn.BatchNorm1d(num_emb)
        self.relu = nn.ReLU(inplace=True)
        self.proxy = torch.nn.Parameter(torch.randn(num_class, num_emb).cuda())
        nn.init.kaiming_normal_(self.proxy, mode='fan_out')
        self.num_class = num_class
        self.temp = temp

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x, getemb=False):
        featwonorm = self.fc1(x)
        featwonorm = self.bn(featwonorm)
        featwonorm = self.relu(featwonorm)
        emb = F.normalize(featwonorm)
        proxy = self.l2_norm(self.proxy)
        cos = F.linear(emb, proxy)
        x_out = cos / self.temp
        if getemb:
            remb = emb
            return x_out, remb
        else:
            return x_out
