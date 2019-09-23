import re
import glob
import torch
import torch.nn as nn
import numpy as np

from model.rnn_related_model import RnnRelatedModel
from data.data_splitter import Splitter
from data.data_generator import DataGenerator
from utils import eval_result


class Trainer(object):
    def __init__(self,
                 # splitter related
                 label_list: list,
                 is_sliced_data: bool,
                 is_create_negative_sample: bool,
                 negative_sample_scale: int,
                 # data generator related
                 sample_spaces: int,
                 # model related
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 learning_rate: float,
                 embedding_scale: int,
                 # train related
                 epochs: int,
                 train_batch: int,
                 eval_batch: int):
        self.epochs = epochs
        self.train_batch = train_batch
        self.eval_batch = eval_batch

        splitter = Splitter(label_list, is_sliced_data, is_create_negative_sample, negative_sample_scale)
        self.nested_label2id = splitter.label2id
        self.generator = DataGenerator(splitter, sample_spaces)
        self.model = RnnRelatedModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=self.generator.get_label_size(),
            # TODO: this should be dynamically generated
            loss_weight=None,
            embedding_scale=embedding_scale
        )
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=learning_rate)

        self.name = "trainer_" + self.generator.name

    def train(self):
        data_gen = self.generator
        model = self.model
        optimizer = self.optimizer

        total_batch = data_gen.get_total_batch(self.train_batch, "train")
        print("---------- Start Training ----------")
        for epoch in range(self.epochs):
            for idx, batch_data in enumerate(data_gen.generate(batch_size=self.train_batch, mode="train")):
                batch_input, batch_label = batch_data
                optimizer.zero_grad()
                preds, loss = model(batch_input, batch_label, "train")
                loss.backward()
                optimizer.step()

                if idx % 10 == 0:
                    print(f"loss for batch {idx} / {total_batch}: {loss.data}")

            self.save_model(epoch + 1)

    def evaluate(self):
        self.restore_model()

        data_gen = self.generator
        model = self.model
        model.eval()
        preds, labels = [], []
        total_batch = data_gen.get_total_batch(self.eval_batch, "eval")
        print("---------- Start Evaluating ----------")
        for idx, batch_data in enumerate(data_gen.generate(batch_size=self.eval_batch, mode="eval")):
            batch_input, batch_label = batch_data
            pred, _ = model(batch_input, batch_label, "eval")
            preds.extend(pred.detach().numpy())
            labels.extend(batch_label)

            if idx % 10 == 0:
                print(f"eval batch: {idx} / {total_batch}")

        metrics = eval_result(preds, labels, self.nested_label2id, True)
        return metrics

    def predict(self):
        self.restore_model()

    def save_model(self, epoch: int):
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   './result/{}.epoch-{}.ckpt'.format(self.name, epoch))

    def restore_model(self, epoch: int = -1):
        if epoch == -1:
            model_file_paths = glob.glob('result/{}.epoch-*'.format(self.name))
            try:
                epoch = max([int(re.findall('{}\.epoch-(\d*)\.ckpt'.format(self.name), path)[0]) for path in model_file_paths])
            except:
                print("----------  Error: Model not saved. ----------")
                exit(-1)
        ckpt_path = './result/{}.epoch-{}.ckpt'.format(self.name, epoch)
        model_ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(model_ckpt['state_dict'])
        self.optimizer.load_state_dict(model_ckpt['optimizer'])


if __name__ == '__main__':
    trainer = Trainer(
        label_list=[1],
        is_sliced_data=False,
        is_create_negative_sample=True,
        negative_sample_scale=1,
        sample_spaces=10,
        input_size=12,
        hidden_size=128,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.9,
        embedding_scale=1,
        epochs=1,
        train_batch=20,
        eval_batch=20
    )
    trainer.train()
    trainer.evaluate()
