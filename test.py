import os, sys
import shutil
import time
import argparse
import torch
import numpy as np

from net.network import Network
from misc import get_logger, seed_all
# from dataset import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data
from dataset import SequentialPointcloudPatchSampler
from mydataset import PointCloudDataset, load_data, PatchDataset


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='')
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--ckpt_dirs', type=str, default='', help='multiple files separated by comma')
    parser.add_argument('--ckpt_iters', type=str, default='', help='multiple files separated by comma')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--eval_list', type=str, nargs='*',
                        help='list of .txt files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--sparse_patches', type=eval, default=True, choices=[True, False],
                        help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_pn', type=eval, default=False, choices=[True, False])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            sparse_patches=args.sparse_patches,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=args.batch_size,
            num_workers=0,
        )
    return test_dset, test_set, test_dataloader


# Arguments
args = parse_arguments()
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
print('Arguments:\n %s\n' % arg_str)

seed_all(args.seed)
PID = os.getpid()

assert args.gpu >= 0, "ERROR GPU ID!"
_device = torch.device('cuda:%d' % args.gpu)

### Datasets and loaders
test_dset, test_set, test_dataloader = get_data_loaders(args)


def normal_RMSE(normal_gts, normal_preds, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()
        # print(out_str)

    rms = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5 = []
    pgp_alpha = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented RMSE
        ####################################################################
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### portion of good points
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))
        pgp_alpha.append(pgp_alpha_shape)

        ### Oriented RMSE
        ####################################################################
        ang_o = np.rad2deg(np.arccos(nn))   # angle error in degree
        ids = ang_o > 90.0
        p = sum(ids) / normal_pred.shape[0]

        ### if more than half of points have wrong orientation, then flip all normals
        if p > 0.5:
            nn = np.sum(np.multiply(normal_gt, -1 * normal_pred), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))    # angle error in degree
            ids = ang_o > 90.0
            p = sum(ids) / normal_pred.shape[0]

        rms_o.append(np.sqrt(np.mean(np.square(ang_o))))

    avg_rms   = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_file.close()

    return avg_rms, avg_rms_o


def test(ckpt_dir, ckpt_iter):
    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_save_dir, exist_ok=True)
    # pc_dir = os.path.join(output_dir, 'sample_%s' % args.data_set)
    # os.makedirs(pc_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=_device)
    model = Network(num_pat=args.patch_size,
                    num_sam=args.sample_size,
                    encode_knn=args.encode_knn,
                ).to(_device)

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = sum([np.prod(p.size()) for p in model_parameters])
    # logger.info('Num_params: %d' % num_params)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of trainable parameters: %d' % trainable_num)

    model.load_state_dict(ckpt)
    model.eval()

    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]

    num_batch = len(test_dataloader)
    normal_prop = torch.zeros([shape_patch_count, 3])

    total_time = 0
    
    lr_start = time.time()
    ds_start = time.time()
    batch_debug_counter = 0
    timings_accumulator = []


    for batchind, batch in enumerate(test_dataloader, 0):
        ds_duration = time.time() - ds_start
        ds_msg = (f"{bcolors.FAIL}{str(ds_duration)}{bcolors.ENDC}" if ds_duration > 2 else str(ds_duration))
        # print("[DEBUG] Step DataLoader: ", ds_msg)

        batch_debug_counter += 1
        # Extend local accumulator with dataset timings
        timings_accumulator.extend(test_set.debug_times)
        # Clear the dataset-side list so we don’t double-collect
        test_set.debug_times = []

        arr = np.array(timings_accumulator)
        means = arr.sum(axis=0)
        print(f"[DEBUG STATS after {batch_debug_counter} batches] "
              f"shape_index: {means[0]:.4f}, make_patch: {means[1]:.4f}, "
              f"build_dict: {means[2]:.4f}, subsample: {means[3]:.4f}, PCA: {means[4]:.4f}")
        # Reset
        timings_accumulator = []


        pcl_pat = batch['pcl_pat'].to(_device)
        data_trans = batch['pca_trans'].to(_device)
        pcl_sample = batch['pcl_sample'].to(_device) if 'pcl_sample' in batch else None

        start_time = time.time()
        with torch.no_grad():
            n_est = model(pcl_pat, pcl_sample=pcl_sample, mode_test=True)
        end_time = time.time()
        elapsed_time = 1000 * (end_time - start_time)
        total_time += elapsed_time

        if batchind % 1 == 0:
            batchSize = pcl_pat.size()[0]
            logger.info('[%d/%d] %s: time per patch: %.3f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], elapsed_time / batchSize))

            # weights = weights.transpose(2, 1)                                 # (B, N, 1)
            # pcl = torch.cat([pcl_pat[:,:model.num_out,:], weights], dim=-1)   # (B, N, 4)
            # normal = pcl_pat[:,0:1,:] + n_est.unsqueeze(1) / 2          # (B, 1, 3)
            # normal = torch.cat([pcl_pat[:,0:1,:], normal], dim=1)       # (B, 2, 3)
            # # pcl = torch.cat([pcl, normal], dim=1)
            # pcl = pcl[0].cpu().detach().numpy()
            # np.savetxt(pc_dir + '/%d_pc.txt' % batchind, pcl, fmt='%.6f')
            # normal = normal[0].cpu().detach().numpy()
            # np.savetxt(pc_dir + '/%d_nor.poly' % batchind, normal, fmt='%.6f')

        if data_trans is not None:
            ### transform predictions with inverse pca rotation (back to world space)
            n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

        ### Save estimated normals to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

            if shape_patches_remaining <= batch_patches_remaining:
                normals_to_write = normal_prop.cpu().numpy()
                # eps=1e-6
                # normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy') # for faster reading speed
                # create folder if not exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # np.save(save_path, normals_to_write)
                if args.save_pn:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals')
                    # np.savetxt(save_path, normals_to_write, fmt='%.6f')
                logger.info('Save normal: {}'.format(save_path))
                logger.info('Total Time: %.2f sec, Shape Num: %d / %d \n' % (total_time/1000, shape_ind+1, shape_num))

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = torch.zeros([shape_patch_count, 3])
        
        # TODO Remove this
        duration = time.time() - lr_start
        msg = (f"{bcolors.WARNING}{str(duration)}{bcolors.ENDC}" if duration > 2 else str(duration))
        # print("[DEBUG] Step Test Process: ", msg)
        lr_start = time.time()
        ds_start = time.time()

        

    logger.info('Total Time: %.2f sec, Shape Num: %d' % (total_time/1000, shape_num))
    return output_dir, file_save_dir


def eval(normal_gt_path, normal_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(eval_summary_dir, exist_ok=True)

    all_avg_rms = []
    all_avg_rms_o = []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list + '.txt')
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shape data of the list
        normal_gts = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape + '/normal', dtype=np.float32)  # (N, 3)
            if os.path.exists(os.path.join(normal_gt_path, shape + '.pidx')):
                points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)      # (n,)
                eval_sparse = True
            else:
                points_idx = np.arange(normal_gt.shape[0])
                eval_sparse = False

            normal_gt = normal_gt[points_idx, :]
            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]
            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

        ### compute RMSE per-list
        avg_rms, avg_rms_o = normal_RMSE(normal_gts=normal_gts,
                            normal_preds=normal_preds,
                            eval_file=os.path.join(eval_summary_dir, cur_list + '_evaluation_results.txt'))
        all_avg_rms.append(avg_rms)
        all_avg_rms_o.append(avg_rms_o)

        print('### RMSE: %f' % avg_rms)
        print('### RMSE_Ori: %f' % avg_rms_o)

    s = '\n {} \n All RMS not oriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_rms), np.mean(all_avg_rms))
    print(s)

    s = '\n {} \n All RMS oriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_rms_o), np.mean(all_avg_rms_o))
    print(s)
    print('eval_sparse:', eval_sparse)

    ### delete the normal files
    if not args.save_pn:
        shutil.rmtree(normal_pred_path)
    return all_avg_rms, all_avg_rms_o



if __name__ == '__main__':
    ckpt_dirs = args.ckpt_dirs.split(',')
    ckpt_iters = args.ckpt_iters.split(',')

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)

        for ckpt_iter in ckpt_iters:
            output_dir, file_save_dir = test(ckpt_dir=ckpt_dir, ckpt_iter=ckpt_iter)
            if not output_dir or args.data_set in ['Semantic3D', 'KITTI_sub', 'WireframePC']:
                continue
            all_avg_rms, all_avg_rms_o = eval(normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                                                normal_pred_path=file_save_dir,
                                                output_dir=output_dir)

            s = '%s: %s | Mean: %f \t|| %s | Mean: %f\n' % (ckpt_iter, str(all_avg_rms), np.mean(all_avg_rms),
                                                                    str(all_avg_rms_o), np.mean(all_avg_rms_o))
            log_file_sum.write(s)
            log_file_sum.flush()
            eval_dict += s

        log_file_sum.close()
        s = '\n All RMS not oriented and oriented (shape average): \n{}\n'.format(eval_dict)
        print(s)


