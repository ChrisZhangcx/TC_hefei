import numpy as np


def eval_result(preds: list, labels: list, label2id: dict, is_squeeze_preds: bool = False) -> dict:
    assert len(preds) == len(labels)

    length = len(preds)
    preds = np.array(preds)
    labels = np.array(labels)
    if is_squeeze_preds:
        preds = np.argmax(preds, axis=-1)

    id2label = {label2id[label]: label for label in label2id.keys()}
    metrics = {idx: [0, 0, 0] for idx in id2label}      # tp, fp, fn
    for i in range(length):
        pred = preds[i]
        label = labels[i]

        if pred == label:
            metrics[pred][0] += 1
        else:
            metrics[pred][1] += 1
            metrics[label][2] += 1

    total_p, total_r = 0.0, 0.0
    res = dict()
    for key in metrics.keys():
        if id2label[key] == -1:
            continue
        tp, fp, fn = metrics[key]
        p = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0.0
        r = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        res[id2label[key]] = [p, r, f1]

        total_p += p
        total_r += r
    total_p /= len(res.keys())
    total_r /= len(res.keys())
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if total_p + total_r > 0 else 0.0

    print(f'---------- P, R, Macro F1 for labels: {total_p, total_r, total_f1} ----------')
    return res


def calc_embedding_dims(feature_size: int, scale: int) -> int:
    return int(np.log(feature_size) * scale)


def stupid_resolution():
    with open("./data/hf_round1_subA.txt", "r", encoding="utf-8") as f:
        res = []
        for line in f.readlines():
            res.append(line[:-1] + "\t" + "窦性心律" + "\t" + "QRS低电压" + "\n")
        with open("./data/results.txt", "w", encoding="utf-8") as r:
            r.writelines(res)


if __name__ == '__main__':
    # stupid_resolution()
    preds = [[0.8, 0.7, 0.4, -0.2, 0.1]]
    labels = [0]
    label2id = {0: 0, 2: 1, 7: 2, 17: 3, -1: 4}
    eval_result(preds, labels, label2id, True)
