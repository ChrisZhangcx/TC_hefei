import os
import pickle
import pandas as pd
import numpy as np
import params


class Parser(object):
    """
    Parse origin data into specified format and save to data folder.
    This parser will only be used once to parse and save parsed data.
    """
    def __init__(self):
        # module name
        self.name = "parser"
        # data path
        self.train_folder = params.TRAIN_FOLDER
        self.test_a_folder = params.TEST_A_FOLDER
        self.label_path = params.LABEL_PATH
        self.submit_a_path = params.SUBMIT_A_PATH
        self.arrythmia_path = params.ARRYTHMIA_PATH
        self.train_file_paths = Parser.scan_folder(self.train_folder)
        self.test_a_file_paths = Parser.scan_folder(self.test_a_folder)
        # transfer arrythmia to id
        self.arrythmia2id = params.ARRYTHMIA2ID
        self.label_size = len(self.arrythmia2id.keys())
        # path to save parsed data
        self.parsed_data_saved_path = params.PARSED_DATA_PATH

    def parse_data(self):
        def parse_label_file(path: str) -> dict:
            def parse_age(age: str):
                try:
                    return int(age)
                except:
                    return -1

            def parse_sex(sex: str):
                if sex.lower() == "male":
                    return 1
                elif sex.lower() == "female":
                    return 0
                else:
                    return -1

            ret = dict()
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    name = line[0].split('.txt')[0]
                    age = parse_age(line[1])
                    sex = parse_sex(line[2])
                    disease = [self.arrythmia2id[sym] for sym in line[3:]]
                    ret[name] = {'age': age, 'sex': sex, 'disease': sorted(disease)}
            return ret

        total_data = list()
        label_data = parse_label_file(self.label_path)
        cnt, total_cnt = 0, len(self.train_file_paths)

        for p in self.train_file_paths:
            tfp = os.path.join(self.train_folder, p)
            name = p.strip().split('.txt')[0]
            label = label_data[name]
            total_data.append({
                "name": name,                                   # str
                # total data will cause memory error
                # "data": Parser.parse_train_file(tfp),         # pd.DataFrame
                "path": tfp,                                    # str
                "age": label['age'],                            # int
                "sex": label['sex'],                            # int: 1-male/0-female/-1-none
                "disease": Parser.label2one_hot(label['disease'], self.label_size)
            })

            cnt += 1
            if cnt % 1000 == 0:
                print(f"{cnt} / {total_cnt} train file parsed.")
        print("train file parse finished.")

        pickle.dump(total_data, open(self.parsed_data_saved_path, 'wb'))

    def parse_test_data(self):
        total_data = list()
        cnt, total_cnt = 0, len(self.test_a_file_paths)

        for p in self.test_a_file_paths:
            tfp = os.path.join(self.test_a_folder, p)

            name = p.strip().split('.txt')[0]
            total_data.append({
                "name": name,                                   # str
                # total data will cause memory error
                # "data": Parser.parse_train_file(tfp),         # pd.DataFrame
                "path": tfp,                                    # str
                "age": label['age'],                            # int
                "sex": label['sex'],                            # int: 1-male/0-female/-1-none
            })

            cnt += 1
            if cnt % 1000 == 0:
                print(f"{cnt} / {total_cnt} train file parsed.")
        print("train file parse finished.")

        pickle.dump(total_data, open(self.parsed_data_saved_path, 'wb'))

    def load_parsed_data(self, is_slice_data: bool) -> list:
        try:
            if is_slice_data:
                return pickle.load(open(self.parsed_data_saved_path + "_sliced", 'rb'))
            else:
                return pickle.load(open(self.parsed_data_saved_path, 'rb'))
        except:
            print("----------- Error! Cannot load parsed data, check if this file exists. -----------")
            exit(-1)

    @staticmethod
    def scan_folder(path: str) -> list:
        file_names = os.listdir(path)
        return file_names

    @staticmethod
    def label2one_hot(label: list, max_label_size: int) -> list:
        ret = [0 for i in range(max_label_size)]
        for l in label:
            ret[l] = 1
        return ret

    @staticmethod
    def parse_train_file(path: str) -> pd.DataFrame:
        train_data = pd.read_csv(path, header=0, sep=' ')
        train_data['III'] = train_data['II'] - train_data['I']
        train_data['aVR'] = - (train_data['I'] + train_data['II']) / 2
        train_data['aVL'] = train_data['I'] - train_data['II'] / 2
        train_data['aVF'] = train_data['II'] - train_data['I'] / 2
        train_data = train_data.astype("int")
        return train_data.values.tolist()

    """
    # this is not used for now
    def generate_batch_data(self, batch_size: int, is_sliced_data: bool):
        data = self.load_parsed_data(is_slice_data=is_sliced_data)
        np.random.shuffle(data)

        total_batch = (len(data) - 1) // batch_size + 1
        for batch_id in range(total_batch):
            begin = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            batch_data = data[begin:end]

            batch_input, batch_label = [], []
            for per_data in batch_data:
                batch_input.append(Parser.parse_train_file(per_data['path']))
                batch_label.append(per_data['disease'])
            yield np.array(batch_input), np.array(batch_label)
    """


if __name__ == '__main__':
    parser = Parser()
    data = parser.load_parsed_data(is_slice_data=True)
    path = data[0]['path']
    d = Parser.parse_train_file(path)
    # parser.parse_data()
    # for idx, data in enumerate(parser.generate_batch_data(batch_size=3, is_sliced_data=True)):
    #     print(idx)
    #     print(data)
