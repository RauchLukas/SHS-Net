

import os 
import numpy as np 


# list all files wiht the file exten in a directory
def list_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith('.' + extension)]


room_id = "room_859972_sub"
file_dir = './dataset/structured3d/test/scene_03250'
file_dir = os.path.join(file_dir, room_id)

file_list = list_files(file_dir, 'npy')

source_data = np.load(os.path.join(file_dir, file_list[0]))

n_samples = 50000 
sample_idx = np.random.choice(source_data.shape[0], n_samples, replace=False)

for file in file_list:
    data = np.load(os.path.join(file_dir, file))
    data = data[sample_idx, :]
    np.save(os.path.join(file_dir, file), data)


print('Subsampling done!')
pass