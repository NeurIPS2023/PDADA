from __future__ import print_function
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim
import numpy as np
from copy import deepcopy
from pathlib import Path

from trainer import base
from model.domain_discriminator import DomainDiscriminator
from loaders.return_dataset import get_loader, get_data_transforms
from utils.forever_data_iter import ForeverDataIterator
from utils.adv_loss import DomainAdversarialLoss
from utils.weights_loader import load_weights


class Trainer(base.Trainer):
    def __init__(self, args):
        super().__init__(args)

        ''' domain adv loss '''
        self.adv_loss = DomainAdversarialLoss(self.domain_discri).cuda()

        ''' resume pretrained src model '''
        assert (os.path.exists(args.resume)), "resume path not exist"
        weight_path = Path(args.resume)
        checkpoint = torch.load(weight_path)
        self.model = load_weights(self.model, checkpoint['model'])
        print("Checkpoint {} loaded!".format(weight_path))

    def resume_model(self, args):
        print("model loading")

    def define_model(self):
        domain_discri = DomainDiscriminator(in_feature=512, hidden_size=1024)
        self.domain_discri = domain_discri.cuda()
        return super().define_model()

    def define_optim(self, args):
        param_groups = self.model.trainable_parameters()
        bblr = args.lr * 0.1
        param_list = [{'params': param_groups[0], 'lr': bblr, 'initial_lr': bblr},
                      {'params': param_groups[1] + list(self.domain_discri.parameters()), 'initial_lr': args.lr}]
        self.optim = optim.SGD(param_list, lr=args.lr, momentum=0.9, nesterov=True,
                               weight_decay=args.weight_decay)

    def Entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def max_discrepancy(self, args, out1, out2):
        previous_pred_cls = out1.argmax()
        now_pred_cls_all = out2.argmax(dim=1)
        if (previous_pred_cls == now_pred_cls_all).sum() < args.ncls:
            is_change = True
        else:
            is_change = False
        return torch.mean(torch.abs(out1 - out2), dim=1).max(), is_change

    def update_proto(self, epoch):
        args = self.args
        num_anno = self.num_annos

        ''' get target feature vectors '''
        self.target_stand_dataset = deepcopy(self.target_dataset)
        self.target_stand_dataset.transform = get_data_transforms(args, transform='test')
        target_stand_loader = get_loader(args, self.target_stand_dataset, False, bs=args.bs, dl=False)

        self.source_stand_dataset = deepcopy(self.source_dataset)
        self.source_stand_dataset.transform = get_data_transforms(args, transform='test')
        source_stand_loader = get_loader(args, self.source_stand_dataset, False, bs=args.bs, dl=False)

        pro_output_all = torch.zeros((len(self.target_stand_dataset), args.ncls))
        feat_all = torch.zeros((len(self.target_stand_dataset), self.inc))
        ad_output_all = torch.zeros((len(self.target_stand_dataset), 1))
        prototype_all = torch.zeros((args.ncls, self.inc))
        each_class_sum = torch.zeros((args.ncls, 1))
        target_stand_loader_iter = iter(target_stand_loader)
        source_stand_loader_iter = iter(source_stand_loader)

        self.model = self.model.cuda()
        self.model.eval()
        self.domain_discri.eval()
        stat = list()
        with torch.no_grad():
            for batch_idx in range(len(source_stand_loader)):
                data_s = next(source_stand_loader_iter)
                im_data_s = data_s[0].cuda()
                label_s = data_s[1]
                _, feat = self.model(im_data_s, getfeat=True)
                feat = feat.cpu()
                for i in range(len(label_s)):
                    prototype_all[int(label_s[i])] += feat[i]
                    each_class_sum[int(label_s[i])] += 1
            prototype_all = prototype_all / each_class_sum

            for batch_idx in range(len(target_stand_loader)):
                data_t = next(target_stand_loader_iter)
                im_data_t = data_t[0].cuda()
                _, feat_ad = self.model(im_data_t, getemb=True)
                output1, feat = self.model(im_data_t, getfeat=True)
                softmax_out = torch.softmax(output1, dim=1)
                ad_out = self.domain_discri(feat_ad)
                if batch_idx == (len(target_stand_loader) - 1):
                    feat_all[batch_idx * args.bs:] = feat.cpu()
                    pro_output_all[batch_idx * args.bs:] = softmax_out.cpu()
                    ad_output_all[batch_idx * args.bs:] = ad_out.cpu()
                else:
                    feat_all[batch_idx * args.bs: (batch_idx + 1) * args.bs] = feat.cpu()
                    pro_output_all[batch_idx * args.bs: (batch_idx + 1) * args.bs] = softmax_out.cpu()
                    ad_output_all[batch_idx * args.bs: (batch_idx + 1) * args.bs] = ad_out.cpu()

            target_stand_loader_iter = iter(target_stand_loader)
            for batch_idx in range(len(target_stand_loader)):
                data_t = next(target_stand_loader_iter)
                labels = data_t[1]
                indexes = data_t[2]
                paths = data_t[3]
                for j in range(len(indexes)):
                    current_index = indexes[j]
                    current_fea = feat_all[current_index]
                    mix_feat_all = (1 - args.lam) * current_fea + args.lam * prototype_all
                    mix_feat_all = mix_feat_all.cuda()
                    mix_pro_all = self.model(mix_feat_all, justclf=True)
                    mix_pro_all = torch.softmax(mix_pro_all, dim=1)
                    mix_pro_all = mix_pro_all.cpu()
                    pbu, is_change = self.max_discrepancy(args, pro_output_all[current_index], mix_pro_all)
                    current_path = paths[j]
                    current_target = labels[j]
                    sim_target = current_fea @ feat_all.T
                    sim_near_target, idx_near_target = torch.topk(sim_target, dim=-1, largest=True, k=args.k + 1)
                    sim_near_target = sim_near_target[1:]
                    idx_near_target = idx_near_target[1:]
                    density_aware_w = sim_near_target.sum() / args.k
                    domainness_w = (1 - ad_output_all[current_index]) / (ad_output_all[current_index] + 1e-7)
                    stat.append([current_path, current_target, current_index, idx_near_target, pbu, density_aware_w,
                                 domainness_w, 0.0, is_change])

            stat = np.array(stat)
            pbu_min = stat[:, 4].min().item()
            pbu_max = stat[:, 4].max().item()
            stat[:, 4] = (stat[:, 4] - pbu_min) / (pbu_max - pbu_min)
            density_aware_w_min = stat[:, 5].min().item()
            density_aware_w_max = stat[:, 5].max().item()
            stat[:, 5] = (stat[:, 5] - density_aware_w_min) / (density_aware_w_max - density_aware_w_min)
            domainness_w_min = stat[:, 6].min().item()
            domainness_w_max = stat[:, 6].max().item()
            stat[:, 6] = (stat[:, 6] - domainness_w_min) / (domainness_w_max - domainness_w_min)
            stat[:, 7] = stat[:, 5] * stat[:, 6]
            stat = sorted(stat, key=lambda x: x[7], reverse=True)
            stat = stat[:int(len(stat) / 2)]

            stat = np.array(stat)
            change_stat_index = list()
            for i in range(len(stat)):
                if stat[i, 8, ...] == True:
                    change_stat_index.append(i)
            stat = stat[change_stat_index]
            stat = sorted(stat, key=lambda x: x[4], reverse=True)

            stat = np.array(stat)
            active_sum = 0
            index = list()
            candicate_ds_index = list()
            for i in range(len(stat)):
                if active_sum == num_anno:
                    break
                candicate_ds_index.append(stat[i, 2, ...])
                index.append(i)
                active_sum = active_sum + 1
            active_samples = stat[index, 0:3, ...]
            print("len_final_stat: {}, len_active_sample: {}".format(len(stat), len(index)))
            active_samples = np.array(active_samples)
            candicate_ds_index = np.array(candicate_ds_index)
            self.labeled_target_dataset.add_item(active_samples[:, 0:2, ...])
            self.target_dataset.remove_item(candicate_ds_index.astype('int64'))
        self.labeled_target_loader = get_loader(args, self.labeled_target_dataset, shuffle=False, dl=False, bs=args.bs,
                                                nw=args.num_workers)
        self.target_loader = get_loader(args, self.target_dataset, shuffle=True, dl=True, bs=args.bs,
                                        nw=args.num_workers)

    def train(self):
        args = self.args
        self.optim.zero_grad()
        self.eval_and_log(epoch=0)
        for epoch in range(args.max_epoch):
            if epoch in self.sampling_epochs:
                self.update_proto(epoch)
            self.train_one_epoch(epoch)
            self.eval_and_log(epoch + 1)

    def train_one_epoch(self, epoch):
        args = self.args
        src_iter = ForeverDataIterator(self.source_loader)
        trg_iter = iter(self.target_loader)
        ltrg_iter = ForeverDataIterator(self.labeled_target_loader)
        self.model.train()
        self.adv_loss.train()

        ''' Train one epoch '''
        for step in range(len(self.target_loader)):
            ''' load S '''
            s_data = next(src_iter)
            s_img, s_gtlbl, s_idx = [i.cuda() for i in s_data[:3]]

            ''' load T '''
            t_data = next(trg_iter)
            t_img = t_data[0].cuda()

            ''' load labeled T '''
            lt_data = next(ltrg_iter)
            lt_img = lt_data[0].cuda()
            lt_gtlbl = lt_data[1].cuda()

            ''' Feedforward S,T '''
            logit, embs = self.model(torch.cat([s_img, t_img]), getemb=True)
            logit_s, logit_t = logit.chunk(2, dim=0)
            embs_s, embs_t = embs.chunk(2, dim=0)
            s_loss_lbl = self.criterion(logit_s, s_gtlbl).mean()
            transfer_loss = self.adv_loss(embs_s, embs_t)
            domain_acc = self.adv_loss.domain_discriminator_accuracy

            ''' Feedforward labeled T '''
            lt_logit, lt_embs = self.model(lt_img, getemb=True)
            lt_loss_lbl = self.criterion(lt_logit, lt_gtlbl).mean()

            ''' Optimize network & log '''
            loss = s_loss_lbl + lt_loss_lbl + transfer_loss
            loss.backward()
            self.am.add({'train-sloss': s_loss_lbl.detach().cpu().item()})
            self.am.add({'train-adv': transfer_loss.detach().cpu().item()})
            self.am.add({'train-tloss': lt_loss_lbl.detach().cpu().item()})
            self.am.add({'train-loss': loss.detach().cpu().item()})
            source_acc = torch.max(logit_s, dim=1)[1].eq(s_gtlbl).float().mean()
            self.am.add({'train-sacc': source_acc.detach().cpu().item()})
            self.am.add({'train-domainacc': domain_acc.detach().cpu().item()})
            self.optim.step()
            self.optim.zero_grad()

            ''' Schedule lr '''
            self.scheduler.step()

            ''' Print current training process '''
            if step % args.log_interval == 0:
                print('[{} ep{}] step{} Loss_S {:.4f} Loss_T {:.4f} Method {}'.format(
                    args.session,
                    epoch,
                    step,
                    self.am.get('train-sloss'),
                    self.am.get('train-tloss'),
                    args.method,
                    args.session))
