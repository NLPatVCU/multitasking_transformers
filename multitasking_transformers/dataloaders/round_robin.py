from typing import List, Tuple
from torch.utils.data import DataLoader
from multi_tasking_transformers.heads import TransformerHead, MaskedLMHead
import torch

class RoundRobinDataLoader():
    """
    A RoundRobinDataLoader takes in a list of DataLoaders and serves batches from them in a round robin fashion.
    Each Dataloader will yield the number based on it's configurations.

    """
    def __init__(self, heads_and_dataloaders: List[Tuple[TransformerHead, DataLoader]], repeat_in_epoch_sampling=True):
        self.heads_and_dataloaders = heads_and_dataloaders
        self.repeat_in_epoch_sampling = repeat_in_epoch_sampling
        self.epoch_batch_counts = {str(head):0 for head, train in self.heads_and_dataloaders}


    def __iter__(self):
        epoch_generators = [(head, enumerate(iter(train))) for head, train in self.heads_and_dataloaders]
        epoch_complete = [False] * len(epoch_generators)
        if len(self.heads_and_dataloaders) > 1 and not any([isinstance(head, MaskedLMHead) for head, _  in self.heads_and_dataloaders]):
            dataset_num_mini_batches = [len(train) for _, train in self.heads_and_dataloaders]
            dataset_with_most_batches = max(range(len(dataset_num_mini_batches)),
                                            key=dataset_num_mini_batches.__getitem__)
        else:
            dataset_with_most_batches = 0

        self.epoch_batch_counts = {str(head): 0 for head, train in self.heads_and_dataloaders}

        # keep sampling batches until the dataset with the most batches is complete.
        while not epoch_complete[dataset_with_most_batches]:
            for i in range(len(epoch_generators)):
                if epoch_complete[dataset_with_most_batches]:
                    break
                head, train_iter = epoch_generators[i]
                try:
                    batch_idx, batch = next(train_iter)
                    self.epoch_batch_counts[str(head)] +=1
                    yield (head, batch_idx, batch)
                except StopIteration:
                    epoch_complete[i] = True
                    if self.repeat_in_epoch_sampling: #resets dataloader to continue sampling from task.
                        epoch_generators[i] = (self.heads_and_dataloaders[i][0],
                                               enumerate(iter(self.heads_and_dataloaders[i][1])))
                        head, train_iter = epoch_generators[i]
                        batch_idx, batch = next(train_iter)
                        yield (head, batch_idx, batch)
