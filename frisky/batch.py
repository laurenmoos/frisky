
from torch.utils.data import IterableDataset
from typing import Callable

from tdigest import TDigest


class Batch(IterableDataset):

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch
        self.digest = TDigest()

    def __iter__(self):
        return self.generate_batch()
        batch = self.generate_batch()
        self.digest.batch_update(batch)

        #TODO: make this configurable
        return self.digest.percentile(15)
