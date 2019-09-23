import pickle


# Constants
FEATURE_NUMBER = 12

ARRYTHMIA_PATH = "data/hf_round1_arrythmia.txt"
FEATURE_VALUE_PATH = "data/feature_value_set.pkl"
TRAIN_FOLDER = "data/train/"
TEST_A_FOLDER = "data/testA/"
LABEL_PATH = "data/hf_round1_label.txt"
SUBMIT_A_PATH = "data/hf_round1_subA.txt"
PARSED_DATA_PATH = "data/parsed_data.pkl"

SPLITTED_DATA_BASE_PATH = "data/splitted/"


ARRYTHMIA2ID = {name: idx for idx, name in
                enumerate([sym.strip() for sym in open(ARRYTHMIA_PATH, "r", encoding="utf-8").readlines()])}
FEATURE_VALUES = pickle.load(open(FEATURE_VALUE_PATH, "rb"))
FEATURE_VALUE2ID = [{value: idx for idx, value in enumerate(FEATURE_VALUES[i])} for i in range(FEATURE_NUMBER)]
FEATURE_SIZES = [len(FEATURE_VALUES[i]) for i in range(FEATURE_NUMBER)]


if __name__ == '__main__':
    pass
