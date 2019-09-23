import numpy as np
import pickle
import os

from data.data_parser import Parser
import params


class Splitter(object):
    """
    Load parsed data and split into valid data according to given label list.

    For example, give label list: [0, 2, 7, 17]. The splitter will check if label in each data
        is 1. If so, label it as positive sample, thus to build a valid data set.

    is_slice_data: whether to use sliced data (2410) or whole data (24106).
    is_create_negative_sample: whether to add negative label samples into returned data set.
    negative_sample_scale: ratio of negative samples to the positive. Only used if is_create_negative_sample is True
    """
    def __init__(self, label_list: list, is_slice_data: bool, is_create_negative_sample: bool, negative_sample_scale: int = 2):
        self.parser = Parser()
        self.name = "split_" + ",".join([str(label) for label in label_list])
        self.save_path = params.SPLITTED_DATA_BASE_PATH + self.name + ".pkl"

        self.label_list = label_list
        self.is_slice_data = is_slice_data
        self.is_create_negative_sample = is_create_negative_sample
        self.negative_sample_scale = negative_sample_scale
        self.label2id = self.re_construct_label2id()
        self.label_count = None

        self.split()

    def split(self):
        data = self.parser.load_parsed_data(self.is_slice_data)
        filtered_data = []
        valid_idx = [0 for i in range(len(data))]

        label_cnt = {label: 0 for label in self.label_list}
        for label in self.label_list:
            for i in range(len(data)):
                if data[i]['disease'][label] == 1:
                    data[i]['label'] = self.label2id[label]
                    valid_idx[i] = 1
                    label_cnt[label] += 1
        filtered_data.extend([data[idx] for idx in range(len(data)) if valid_idx[idx] == 1])

        print(f"--------- count for split label samples: ---------")
        for label in label_cnt:
            print(f"label idx: {label}, label count: {label_cnt[label]}")

        # split: 0.8 - train, 0.2 - eval
        np.random.shuffle(filtered_data)
        filtered_train_num = int(len(filtered_data) * 0.8)
        train_data = filtered_data[:filtered_train_num]
        eval_data = filtered_data[filtered_train_num:]

        # if do negative sampling, try to add negative samples until get enough samples or iterated all data
        if self.is_create_negative_sample:
            negative_samples = [data[idx] for idx in range(len(data)) if valid_idx[idx] == 0]
            for i in range(len(negative_samples)):
                negative_samples[i]['label'] = self.label2id[-1]

            np.random.shuffle(negative_samples)
            sample_num = len(filtered_data) * self.negative_sample_scale
            train_negative = negative_samples[:int(sample_num * 0.8)]
            eval_negative = negative_samples[int(sample_num * 0.8):sample_num]
            print(f"{len(train_negative)} train and {len(eval_negative)} eval negative samples added")
            train_data.extend(train_negative)
            eval_data.extend(eval_negative)

        self.label_count = label_cnt
        self.save({"train": train_data, "eval": eval_data})

    def re_construct_label2id(self) -> dict:
        label2id = {label: idx for idx, label in enumerate(self.label_list)}
        if self.is_create_negative_sample:
            label2id[-1] = len(label2id.keys())     # not belong to either of these label
        return label2id

    def check_data_existence(self):
        return os.path.exists(self.save_path)

    def save(self, data: dict):
        pickle.dump(data, open(self.save_path, 'wb'))

    def load(self) -> dict:
        try:
            return pickle.load(open(self.save_path, 'rb'))
        except:
            print("---------- Error! Cannot load split data. ----------")
            exit(-1)

    def get_label_size(self) -> int:
        return len(self.label2id)


if __name__ == '__main__':
    splitter = Splitter(
        label_list=[0, 2, 7, 17],
        is_slice_data=True,
        is_create_negative_sample=True
    )
    splitter.split()
