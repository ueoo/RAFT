import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import core.datasets as datasets
import evaluate
from core.raft_tiny import RAFTTiny
from core.raft import RAFT
from core.pwc_tiny import PWCTiny


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremely large displacements
MAX_FLOW = 400



def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def multi_scale_loss(flow_preds, flow_gt, valid, weights=[0.5, 0.32, 0.08, 0.02, 0.01], gamma=0.8, max_flow=MAX_FLOW):
    assert len(flow_preds) == len(weights) == 5  # 0, 2, 3, 4, 5
    levels = [0] + [i for i in range(2, 6)]

    flow_loss = 0.0
    _, _, h, w = flow_gt.shape
    for level, flow_pred, weight in zip(levels, flow_preds, weights):
        level_size = (h // (2**level), w // (2**level))
        scale_flow_gt = F.interpolate(flow_gt / 20.0, size=level_size, mode='bilinear', align_corners=True)
        scale_valid = F.interpolate(valid / 20.0, size=level_size, mode='bilinear', align_corners=True)

        scale_valid = scale_valid.squeeze(1)
        mag = torch.sum(scale_flow_gt**2, dim=1).sqrt()
        scale_valid = (scale_valid >= 0.5) & (mag < (max_flow / 20.0))
        scale_valid = scale_valid.unsqueeze(1)
        loss = (flow_pred - scale_flow_gt).abs()
        flow_loss += weight * (scale_valid * loss).mean()

    valid = valid.squeeze(1)
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    epe = torch.sum((flow_preds[0] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, summary_freq, log_dir):
        self.model = model
        self.scheduler = scheduler
        self.summary_freq = summary_freq
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.log_dir = log_dir

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.summary_freq for k in sorted(self.running_loss.keys())]
        training_str = f"[step: {self.total_steps+1:6d}, lr: {self.scheduler.get_last_lr()[0]:10.7f}] "
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.summary_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.summary_freq == self.summary_freq-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def build_model(args):
    if args.model == 'raft':
        model = RAFT(args)
    elif args.model == 'raft_tiny':
        model = RAFTTiny(args)
    elif args.model == 'pwc_tiny':
        model = PWCTiny(args)
    else:
        raise NotImplementedError
    return nn.DataParallel(model, device_ids=args.gpus)


def resize(image1, image2, flow, valid, div=32):
    new_size_h = (image1.shape[2] // div) * div
    new_size_w = (image1.shape[3] // div) * div
    image1 = F.interpolate(image1, (new_size_h, new_size_w), mode='bilinear', align_corners=False)
    image2 = F.interpolate(image2, (new_size_h, new_size_w), mode='bilinear', align_corners=False)
    flow = F.interpolate(flow, (new_size_h, new_size_w), mode='bilinear', align_corners=False)
    valid = valid.unsqueeze(1)
    valid = F.interpolate(valid, (new_size_h, new_size_w), mode='bilinear', align_corners=False)
    # valid = valid.squeeze(1)
    return image1, image2, flow, valid


def train(args):
    model = build_model(args)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    checkpoint_dir = os.path.join(args.output_dir, args.model+"_default")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args.summary_freq, checkpoint_dir)

    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn_like(image1)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn_like(image2)).clamp(0.0, 255.0)
            image1, image2, flow, valid = resize(image1, image2, flow, valid, div=32)

            flow_predictions = model(image1, image2, iters=args.iters)

            if args.model == 'pwc_tiny':
                loss, metrics = multi_scale_loss(flow_predictions, flow, valid, gamma=args.gamma)
            else:
                loss, metrics = sequence_loss(flow_predictions, flow, valid, gamma=args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % args.val_freq == args.val_freq - 1:
                checkpoint_file = os.path.join(checkpoint_dir, f'{total_steps+1}_{args.name}.pth')
                torch.save(model.state_dict(), checkpoint_file)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    checkpoint_file = os.path.join(checkpoint_dir, f'{args.name}.pth')
    torch.save(model.state_dict(), checkpoint_file)

    return checkpoint_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--model', type=str, default='raft')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--summary_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--output_dir', type=str, default='raft_output')
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # torch.manual_seed(1234)
    # np.random.seed(1234)

    train(args)
