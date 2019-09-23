import numpy as np
import pandas as pd
import pickle

from data.data_splitter import Splitter
from data.data_parser import Parser


def parse_train_file(path: str) -> pd.DataFrame:
    train_data = pd.read_csv(path, header=0, sep=' ')
    train_data['III'] = train_data['II'] - train_data['I']
    train_data['aVR'] = - (train_data['I'] + train_data['II']) / 2
    train_data['aVL'] = train_data['I'] - train_data['II'] / 2
    train_data['aVF'] = train_data['II'] - train_data['I'] / 2
    train_data = train_data.astype("int")
    return train_data.values.tolist()


def draw(df: pd.DataFrame):
    fig = plt.figure(figsize=(30, 8 * len(df.columns)))
    for i, item in enumerate(df.columns):
        ax = plt.subplot(len(df.columns), 1, i + 1)
        ax.set_title(item)
        ax.plot(df[item].index, df[item].values)


def count_feature_sizes():
    parser = Parser()
    data = parser.load_parsed_data(is_slice_data=False)
    feature_sizes = {idx: set() for idx in range(12)}
    cnt = 0
    for per in data:
        tpd = np.array(parse_train_file(per['path']))
        for idx in range(12):
            [feature_sizes[idx].add(i) for i in set(tpd[:, idx])]
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
    print([len(feature_sizes[i]) for i in range(12)])
    print([sorted(feature_sizes[i])[0] for i in range(12)])
    pickle.dump(feature_sizes, open('data/feature_value_set.pkl', 'wb'))


if __name__ == '__main__':
    count_feature_sizes()
