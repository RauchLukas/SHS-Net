import os
import sys
import argparse
import torch
import torch.utils.data
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import wandb  ### W&B ADDED ###

from misc import *
from net.network import Network
from dataset import SequentialPointcloudPatchSampler
from mydataset import PointCloudDataset, load_data, PatchDataset, RandomPointcloudPatchSampler


def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Training
    parser.add_argument('--gpu', type=int, default=0)  # This won't matter much with DataParallel
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--scheduler_epoch', type=int, nargs='+', default=[400,600,800])
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--nepoch', type=int, default=800)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='PCPNet')
    parser.add_argument('--trainset_list', type=str, default='')
    parser.add_argument('--valset_list', type=str, default='')   ### W&B ADDED: for validation
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='The number of patches sampled from each shape in an epoch')
    parser.add_argument('--project_name', type=str, default='MyProject')  ### W&B ADDED
    parser.add_argument('--run_name', type=str, default='')               ### W&B ADDED
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    ### TRAIN DATASET
    train_dset = PointCloudDataset(
        root=args.dataset_root,
        mode='train',
        data_set=args.data_set,
        data_list=args.trainset_list,
    )
    train_set = PatchDataset(
        datasets=train_dset,
        patch_size=args.patch_size,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    train_datasampler = RandomPointcloudPatchSampler(train_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_datasampler,
        batch_size=args.batch_size,
        num_workers=int(args.num_workers),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    ### VALIDATION DATASET (optional)
    val_dataloader = None
    if args.valset_list:
        val_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='val',
            data_set=args.data_set,
            data_list=args.valset_list,
        )
        val_set = PatchDataset(
            datasets=val_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        # You can do sequential sampling for validation
        val_sampler = RandomPointcloudPatchSampler(val_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=int(args.num_workers),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    assert len(train_dataloader) > 0, 'Training dataloader is empty!'
    assert len(val_dataloader) > 0, 'Validation dataloader is empty!'
    return train_dataloader, train_datasampler, val_dataloader


def scheduler_fun():
    pre_lr = optimizer.param_groups[0]['lr']
    current_lr = pre_lr * args.lr_gamma
    if current_lr < args.lr_min:
        current_lr = args.lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    logger.info('Update learning rate: %f => %f \n' % (pre_lr, current_lr))


global_step = 0  # track total steps across epochs

def train_one_epoch(epoch):
    global global_step
    model.train()
    total_loss = 0.0
    total_count = 0

    for train_batchind, batch in enumerate(train_dataloader, 0):
        pcl_pat = batch['pcl_pat'].cuda()
        normal_pat = batch['normal_pat'].cuda()
        normal_center = batch['normal_center'].cuda().squeeze()
        pcl_sample = batch['pcl_sample'].cuda() if 'pcl_sample' in batch else None

        optimizer.zero_grad()
        # Forward pass
        pred_point, weights, pred_neighbor = model(pcl_pat, pcl_sample=pcl_sample)

        # -- If using DataParallel, we often call the underlying .module on the model
        if isinstance(model, torch.nn.DataParallel):
            loss, loss_tuple = model.module.get_loss(
                q_target=normal_center,
                q_pred=pred_point,
                ne_target=normal_pat,
                ne_pred=pred_neighbor,
                pred_weights=weights,
                pcl_in=pcl_pat
            )
        else:
            loss, loss_tuple = model.get_loss(
                q_target=normal_center,
                q_pred=pred_point,
                ne_target=normal_pat,
                ne_pred=pred_neighbor,
                pred_weights=weights,
                pcl_in=pcl_pat
            )

        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        bs = pcl_pat.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        # -- Weights & Biases (batch-wise) logging
        wandb.log(
            {
                'train_loss_batch': loss.item(),
                'grad_norm': orig_grad_norm,
                'epoch': epoch
            },
            step=global_step
        )
        global_step += 1

        # Print every N batches
        if (train_batchind + 1) % 10 == 0:
            logger.info(
                '[Train] [Ep %03d: batch %03d/%03d] | '
                'Loss: %.6f | Grad: %.6f'
                % (
                    epoch,
                    train_batchind,
                    train_num_batch - 1,
                    loss.item(),
                    orig_grad_norm
                )
            )

    avg_loss = total_loss / float(total_count) if total_count > 0 else 0.0
    return avg_loss


def validate(model, val_dataloader, device, logger=None):
    if val_dataloader is None:
        return None  # no validation set
    model.eval()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for batchind, batch in enumerate(val_dataloader):
            pcl_pat = batch['pcl_pat'].to(device)
            normal_pat = batch['normal_pat'].to(device)
            normal_center = batch['normal_center'].to(device).squeeze()
            pcl_sample = batch['pcl_sample'].to(device) if 'pcl_sample' in batch else None

            pred_point, weights, pred_neighbor = model(pcl_pat, pcl_sample=pcl_sample)
            loss, loss_tuple = model.get_loss(
                q_target=normal_center,
                q_pred=pred_point,
                ne_target=normal_pat,
                ne_pred=pred_neighbor,
                pred_weights=weights,
                pcl_in=pcl_pat,
            )
            bs = pcl_pat.size(0)
            sum_loss += loss.item() * bs
            count += bs

    avg_loss = sum_loss / float(count) if count > 0 else 0.0
    if logger:
        logger.info(f'[Val] Validation Loss: {avg_loss:.6f}')
    else:
        print(f'[Val] Validation Loss: {avg_loss:.6f}')
    return avg_loss


if __name__ == '__main__':
    args = parse_arguments()
    seed_all(args.seed)

    ### Initialize W&B (before model/dataloaders so config is saved) ###
    wandb.init(
        project=args.project_name,
        name=args.run_name if args.run_name else None,
        config=vars(args)  # store the argparse config
    )
    ### W&B ADDED ###

    # In DataParallel mode, we no longer specify a single GPU device like _device = torch.device('cuda:0')
    # Instead, we just do .cuda() and let DataParallel handle distribution across all visible GPUs.

    PID = os.getpid()

    ### Model
    print('Building model ...')
    model = Network(num_pat=args.patch_size,
                    num_sam=args.sample_size,
                    encode_knn=args.encode_knn,
                    ).cuda()

    # Wrap model in DataParallel
    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs...')
        model = torch.nn.DataParallel(model)

    ### Datasets and loaders
    print('Loading datasets ...')
    train_dataloader, train_datasampler, val_dataloader = get_data_loaders(args)
    print("[DEBUG] val_dataloader len: ", len(val_dataloader) if val_dataloader else 0)
    train_num_batch = len(train_dataloader)

    ### Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #### Logging
    if args.logging:
        log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                                postfix='_' + args.tag if args.tag is not None else '')
        sub_log_dir = os.path.join(log_path, 'log')
        os.makedirs(sub_log_dir)
        logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
        ckpt_dir = os.path.join(log_path, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        logger = get_logger('train', None)

    refine_epoch = -1
    if args.resume != '':
        assert os.path.exists(args.resume), 'ERROR path: %s' % args.resume
        logger.info('Resume from: %s' % args.resume)
        # ...
        # (Same checkpoint loading logic as before)
        # If you need to load partial state dict, etc.
        # Make sure to do .cuda() or move to CPU appropriately
        # or do  model.module.load_state_dict(...) if using DataParallel
        # ...
        logger.info('Load pretrained mode: %s' % args.resume)

    if args.logging:
        code_dir = os.path.join(log_path, 'code')
        os.makedirs(code_dir, exist_ok=True)
        os.system('cp %s %s' % ('*.py', code_dir))
        os.system('cp -r %s %s' % ('net', code_dir))

    ### Arguments
    logger.info('Command: {}'.format(' '.join(sys.argv)))
    arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    logger.info('Arguments:\n' + arg_str)
    logger.info(repr(model))
    logger.info('training set: %d patches (in %d batches)' % (len(train_datasampler), len(train_dataloader)))

    logger.info('Start training ...')
    try:
        best_val_loss = float('inf')
        for epoch in range(refine_epoch+1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)

            start_time = time.time()
            train_loss = train_one_epoch(epoch)
            end_time = time.time()
            logger.info('Training epoch time: %.1f s' % (end_time - start_time))

            # Validation
            val_loss = validate(model, val_dataloader, 'cuda', logger=logger)  # or pass model.module

            # W&B LOG
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss if val_loss is not None else 0.0
            })
            ### W&B ADDED ###

            # Update scheduler if needed
            if epoch in args.scheduler_epoch:
                scheduler_fun()

            # Save best or periodic checkpoint
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.logging:
                    best_ckpt = os.path.join(ckpt_dir, 'ckpt_best.pt')
                    torch.save(model.state_dict(), best_ckpt)
                    logger.info(f'[Val] New best model saved at epoch {epoch} (loss={val_loss:.6f})')

            if epoch % args.interval == 0 or epoch == args.nepoch:
                if args.logging:
                    model_filename = os.path.join(ckpt_dir, f'ckpt_{epoch}.pt')
                    torch.save(model.state_dict(), model_filename)
                    logger.info(f'Model saved: {model_filename}')

    except KeyboardInterrupt:
        logger.info('Terminating ...')
