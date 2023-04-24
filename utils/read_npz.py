import numpy as np

data_file = '../data/xxx.npz'
out = np.load(data_file)

print(out.f.train_images)
print(out.f.train_images.shape)
