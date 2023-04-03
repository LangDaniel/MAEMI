# modified version of: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py 
import argparse
import datetime
import json
import numpy as np
# TODO: replace os with pathlib
import os
import time
from pathlib import Path
import yaml

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"  # version check
from timm.optim import optim_factory

from .util import misc
from .util.misc import NativeScalerWithGradNormCount as NativeScaler

from .util.io_manager import get_output_folder, copy_files, Args, get_param_files
from . import models_mae
from .engine_pretrain import train_one_epoch

def main(args, dataset_train):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    print(args.model)
    model = models_mae.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        norm_pix_loss=args.norm_pix_loss,
        in_chans=args.in_chans,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume and not args.load_partial:
        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)
    if args.resume and args.load_partial:
        misc.load_model_partial(args=args, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs) and (epoch > 0):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

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
    pargs = parser.parse_args()

    def train_model(par_file):
        with open (par_file, 'r') as ff:
            par = ff.read()
            args = yaml.safe_load(par)

        train_data_args = Args({**args['data']['general'], **args['data']['train']})
        dataset_train = seq.CustomDataset(train_data_args)
        val_data_args = Args({**args['data']['general'], **args['data']['valid']})
        dataset_val = seq.CustomDataset(val_data_args)

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
            models_mae.__file__,
            [par_file, 'parameter.yml'],
            './mae3d/util/io_manager.py'
        ]
        copy_files(files_to_copy, model_args.output_dir)
        main(model_args, dataset_train, dataset_val)

    def get_param_files(path):
        path = Path(path)
        if path.is_dir():
            files = sorted([ff for ff in path.iterdir() if ff.suffix == '.yml'])
        elif path.suffix == '.yml':
            files = [path]
        else:
            raise ValueError('non valid parameter files (must be folder or .yml)')
        return files


    for paramf in get_param_files(pargs.pdir):
        print('processing: '+str(paramf))
        try:
            train_model(paramf)
        except Exception as e:
            print(f'failed: {e}')
