# modified version of: https://github.com/facebookresearch/mae/blob/main/main_linprobe.py

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

from .engine_finetune import train_one_epoch, evaluate
from .util.io_manager import get_output_folder, copy_files, Args, get_param_files

from . import sequence as seq

from .util import misc
from .util.pos_embed import interpolate_pos_embed
from .util.misc import NativeScalerWithGradNormCount as NativeScaler
from .util.lars import LARS
from .util.crop import RandomResizedCrop

from . import models_vit

def main(args, dataset_train, dataset_val): 
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()
    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
    )


    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model, args.orig_shape)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss

    if args.train_weights:
        train_weights = torch.tensor(args.train_weights, dtype=torch.float)
        train_weights = train_weights.to(device, non_blocking=True)
    else:
        train_weights = None
    train_criterion = criterion(weight=train_weights)

    if args.val_weights:
        val_weights = torch.tensor(args.val_weights, dtype=torch.float)
        val_weights = val_weights.to(device, non_blocking=True)
    else:
        val_weights = None
    val_criterion = criterion(weight=val_weights)

    print("train criterion = %s" % str(train_criterion))
    print("val criterion = %s" % str(val_criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, val_criterion, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    min_test_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, val_criterion, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        test_loss = test_stats['loss']

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            print(f'Epoch with min loss {test_loss:.6} -> saving to disk')
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, name='best_loss.pth')


        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    from . import sequence as seq 

    parser = argparse.ArgumentParser()
    parser.add_argument('pdir', help='parameter directory')
    parser.add_argument('--gpu', help='choose gpu [0, 1]')
    pargs = parser.parse_args()

    def train_model(par_file):
        with open (par_file, 'r') as ff:
            par = ff.read()
            args = yaml.safe_load(par)
        train_data_args = Args({**args['data']['general'], **args['data']['train']})
        val_data_args = Args({**args['data']['general'], **args['data']['valid']})
        model_args = Args({**args['model'], **args['training']})
    
        model_args.output_dir = get_output_folder(
            args['output']['output_dir'],
            args['output']['sub'],
        )
        model_args.log_dir = model_args.output_dir
    
        # sequence output size = model input size
        model_args.input_size = train_data_args.shape
    
        if model_args.output_dir:
            Path(model_args.output_dir).mkdir(parents=True, exist_ok=True)
        files_to_copy = [
            __file__, seq.__file__,
            models_vit.__file__,
            [par_file, 'parameter.yml'],
            './mae3d/util/io_manager.py'
        ]
        copy_files(files_to_copy, model_args.output_dir)
        main(model_args, train_data_args, val_data_args)
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('pdir', help='parameter directory')
        pargs = parser.parse_args()
        
        for paramf in get_param_files(pargs.pdir):
            print('processing: '+str(paramf))
            train_model(paramf)
