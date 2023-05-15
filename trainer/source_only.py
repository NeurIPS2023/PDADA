from __future__ import print_function
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
from trainer import base


class Trainer(base.Trainer):
    def __init__(self, args):
        super().__init__(args)

    def train_one_epoch(self, epoch):
        args = self.args
        src_iter = iter(self.source_loader)
        pbar = tqdm(src_iter)
        self.model.train()

        for step, t_data in enumerate(pbar):
            try:
                s_data = next(src_iter)
            except(StopIteration):
                src_iter = iter(self.source_loader)
                s_data = next(src_iter)
            s_img, s_gtlbl, s_idx = [i.cuda() for i in s_data[:3]]

            s_logit, s_embs = self.model(s_img, getemb=True)
            loss_lbl = self.criterion(s_logit, s_gtlbl).mean()

            loss = loss_lbl
            loss.backward()
            self.am.add({'train-sloss': loss_lbl.detach().cpu().item()})
            self.am.add({'train-loss': loss.detach().cpu().item()})
            source_acc = torch.max(s_logit, dim=1)[1].eq(s_gtlbl).float().mean()
            self.am.add({'train-sacc': source_acc.detach().cpu().item()})
            self.optim.step()
            self.optim.zero_grad()

            self.scheduler.step()

            if step % args.log_interval == 0:
                print('[{} ep{}] step{} Loss_S {:.4f} Method {}'.format(
                    args.session,
                    epoch,
                    step,
                    self.am.get('train-sloss'),
                    args.method,
                    args.session))
