import numpy as np
import torch.nn as nn

from params import FEATURE_VALUE2ID, FEATURE_NUMBER
from data.data_parser import Parser
from data.data_splitter import Splitter


class DataGenerator(object):
    def __init__(self, splitter: Splitter, sample_space: int):
        self.label_size = splitter.get_label_size()
        self.data = splitter.load()
        self.name = splitter.name
        self.total_batch = 0
        self.sample_space = sample_space

    def generate(self, batch_size: int, mode: str):
        data = self.data[mode]
        np.random.shuffle(data)

        total_batch = (len(data) - 1) // batch_size + 1
        for batch_id in range(total_batch):
            begin = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            batch_data = data[begin: end]

            batch_input, batch_label = [], []
            for per_data in batch_data:
                features = np.array(Parser.parse_train_file(per_data['path'])[::self.sample_space])
                # for col in range(features.shape[1]):
                #     for row in range(features.shape[0]):
                #         features[row, col] = FEATURE_VALUE2ID[col][features[row, col]]
                batch_input.append(features)
                batch_label.append(per_data['label'])
            yield np.array(batch_input), np.array(batch_label)

    def get_label_size(self) -> int:
        return self.label_size

    def get_total_batch(self, batch_size: int, mode: str) -> int:
        return (len(self.data[mode]) - 1) // batch_size + 1


if __name__ == '__main__':
    splitter = Splitter(
        label_list=[0, 2, 7, 17],
        is_slice_data=False,
        is_create_negative_sample=True
    )
    splitter.split()
    generator = DataGenerator(splitter, 10)
    for idx, data in enumerate(generator.generate(batch_size=20)):
        print(idx, data[0].shape, data[1].shape)
