import numpy as np
import random
import torch
import torchvision



class BSR_transformer():
    def __init__(self, num_block=2, deg = 24):
        super().__init__()
        self.num_block = num_block
        self.deg = deg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def rot(self, x):
        return torchvision.transforms.RandomRotation(self.deg, fill=0)(x)

    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        if length in rand_norm:
            rand_norm[rand_norm.argmax()] -= 10
            rand_norm[rand_norm.argmin()] += 10
        if 0 in rand_norm:
            rand_norm[rand_norm.argmax()] -= 10
            rand_norm[rand_norm.argmin()] += 10
        #print(rand_norm)
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def shuffle(self, x):
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(list(self.rot(sub_x) for sub_x in self.shuffle_single_dim(x_strip, dim=dims[1])), dim=dims[1]) for x_strip in x_strips],
                         dim=dims[0])

    def transform(self, x, num_scale=1):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(num_scale)])

