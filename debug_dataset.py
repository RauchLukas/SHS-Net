import os
from mydataset import PointCloudDataset


dataset_root = "./dataset/"
data_set = "structured3d"


data_list = "test_samples"


sparse_patches = False

test_dset = PointCloudDataset(
            root=dataset_root,
            mode='test',
            data_set=data_set,
            data_list=data_list,
            sparse_patches=sparse_patches,
        )


