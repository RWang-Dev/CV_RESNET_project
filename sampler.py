from torch.utils.data import Sampler
import math, random

class SmallObjectSampler(Sampler):
    def __init__(self, dataset, batch_size=4, small_fraction=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.small_fraction = small_fraction
        
        self.small_indices = []
        self.normal_indices = []
        
        for i in range(len(dataset)):
            img, target = dataset[i]
            if len(target['boxes']) > 0:
                areas = (target['boxes'][:, 2] - target['boxes'][:, 0]) * (target['boxes'][:, 3] - target['boxes'][:, 1])
                if (areas < 500).any():
                    self.small_indices.append(i)
                else:
                    self.normal_indices.append(i)
            else:
                self.normal_indices.append(i)

        random.shuffle(self.small_indices)
        random.shuffle(self.normal_indices)

    def __iter__(self):
        small_count = math.ceil(self.batch_size * self.small_fraction)
        normal_count = self.batch_size - small_count
        
        small_pool = self.small_indices[:]
        normal_pool = self.normal_indices[:]
        
        while len(small_pool) >= small_count and len(normal_pool) >= normal_count:
            batch = []
            for _ in range(small_count):
                batch.append(small_pool.pop())
            for _ in range(normal_count):
                batch.append(normal_pool.pop())
            yield from batch

    def __len__(self):
        small_count = math.ceil(self.batch_size * self.small_fraction)
        normal_count = self.batch_size - small_count
        
        max_batches = min(len(self.small_indices)//small_count, len(self.normal_indices)//normal_count)
        return max_batches * self.batch_size