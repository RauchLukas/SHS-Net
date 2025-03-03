import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='', choices=['train', 'test'])
parser.add_argument('--data_set', type=str, default='PCPNet',
                    choices=['PCPNet', 'FamousShape', 'FamousShape3k', 'FamousShape5k', 'FamousShape50k', 'WireframePC', 'SceneNN', 'NestPC', 'Semantic3D', 'KITTI_sub', 'structured3d'])
parser.add_argument('--ckpt_dirs', type=str, default='001')
parser.add_argument('--ckpt_iters', type=str, default='800')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--project_name', type=str, default='MyProject')  ### W&B ADDED
FLAGS = parser.parse_args()

dataset_root = './dataset/'
gpu = FLAGS.gpu
num_workers = FLAGS.num_workers
lr = 0.0009
encode_knn = 16
sample_size = 1200
train_patch_size = 700
train_batch_size = 145

project_name = FLAGS.project_name  ### W&B ADDED

if FLAGS.mode == 'train':
    dataset = FLAGS.data_set
    trainset_list = 'train_samples'
    valset_list = 'val_samples'
    resume = FLAGS.resume

    os.system('CUDA_VISIBLE_DEVICES={} python mytrain.py --dataset_root={} --data_set={} --trainset_list={} --valset_list={} --patch_size={} --batch_size={} \
                                                    --sample_size={} --encode_knn={} --lr={} --resume={} --project_name={}'.format(
                gpu, dataset_root, dataset, trainset_list, valset_list, train_patch_size, train_batch_size, sample_size, encode_knn, lr, resume, project_name))
 
elif FLAGS.mode == 'test':
    tag = ''
    log_root = './log/'
    data_set = FLAGS.data_set
    test_patch_size = train_patch_size
    test_batch_size = 700
    ckpt_dirs = FLAGS.ckpt_dirs
    if ckpt_dirs == '':
        ckpt_dirs = os.path.split(os.path.abspath(os.path.dirname(os.getcwd())))[-1]
    ckpt_iters = FLAGS.ckpt_iters

    save_pn = True           # to save the point normals as '.normals' file
    sparse_patches = False   # to output sparse point normals or not

    testset_list = None
    eval_list = None
    if data_set == 'PCPNet':
        testset_list = 'testset_PCPNet'
        eval_list = 'testset_no_noise testset_low_noise testset_med_noise testset_high_noise \
                    testset_vardensity_striped testset_vardensity_gradient'
    elif data_set == 'FamousShape':
        testset_list = 'testset_FamousShape'
        eval_list = 'testset_noise_clean testset_noise_low testset_noise_med testset_noise_high \
                    testset_density_stripe testset_density_gradient'
    elif data_set in ['FamousShape3k', 'FamousShape5k', 'FamousShape50k']:
        testset_list = 'testset_%s' % data_set
        eval_list = testset_list
    elif data_set == 'SceneNN':
        testset_list = 'testset_SceneNN'
        eval_list = 'testset_SceneNN_clean testset_SceneNN_noise'
    elif data_set == 'Semantic3D':
        testset_list = 'testset_Semantic3D'
        eval_list = testset_list
    elif data_set == 'KITTI_sub':
        testset_list = 'testset_KITTI0608'
        eval_list = testset_list
    elif data_set == 'WireframePC':
        testset_list = 'testset_WireframePC'
        eval_list = testset_list
        test_patch_size = 200
        sample_size = 200
        tag = '%s-%s' % (test_patch_size, sample_size)
    elif data_set == 'NestPC':
        testset_list = 'testset_NestPC'
        eval_list = testset_list
        test_patch_size = 700
        sample_size = 50
        tag = '%s-%s' % (test_patch_size, sample_size)
    elif data_set == 'structured3d':
        testset_list = 'test_samples'
        eval_list = testset_list

    command = 'python test.py --gpu={} --dataset_root={} --data_set={} --log_root={} --ckpt_dirs={} --ckpt_iters={} --patch_size={} --batch_size={} \
                                --sample_size={} --encode_knn={} --save_pn={} --sparse_patches={} --tag={} --num_workers={}'.format(
            gpu, dataset_root, data_set, log_root, ckpt_dirs, ckpt_iters, test_patch_size, test_batch_size, sample_size, encode_knn, save_pn, sparse_patches, tag, num_workers)

    os.system('{} --testset_list={} --eval_list {}'.format(command, testset_list, eval_list))

else:
    print('The mode is unsupported!')