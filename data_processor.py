import torch
import pandas as pd
import json
from random import sample


def data_loader(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


class CustomDataset():
    def __init__(self, human_path, machine_path, human_path2, machine_path2, train=True):
        human_data = data_loader(human_path)
        machine_data = data_loader(machine_path)
        human_data2 = data_loader(human_path2)
        machine_data2 = data_loader(machine_path2)

        # concatenate prompt and text
        hm_con = [h['prompt'] + h['txt'] for h in human_data]
        ma_con = [m['prompt'] + m['txt'] for m in machine_data]
        hm_con2 = [h['prompt'] + h['txt'] for h in human_data2]
        ma_con2 = [m['prompt'] + m['txt'] for m in machine_data2]

        # trainset or testset
        total_data = list()
        if train:
            total_data += (sample(hm_con, 3500) + hm_con2 + ma_con + sample(ma_con2, 100))
        else:
            total_data += (sample(hm_con, 450) + sample(hm_con2, 50) + sample(ma_con, 450) + sample(ma_con2, 50))

        max_len = 700  # can be chanegd after

        # padding
        for i in range(len(total_data)):
            if len(total_data[i]) < max_len:
                total_data[i] += [0] * (max_len - len(total_data[i]))
            elif len(total_data[i]) > max_len:
                total_data[i] = total_data[i][:max_len]

        if train:
            lab = [1] * 3600 + [0] * 3600
        else:
            lab = [1] * 500 + [0] * 500

        self.x = torch.LongTensor(total_data)
        self.y = lab
        self.n_samples = len(total_data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class TestSet():
    def __init__(self, data_path):
        data = data_loader(data_path)

        # concatenate prompt and text
        total_data = [d['prompt'] + d['txt'] for d in data]

        max_len = 500  # can be chanegd after

        # padding
        for i in range(len(total_data)):
            if len(total_data[i]) < max_len:
                total_data[i] += [0] * (max_len - len(total_data[i]))
            elif len(total_data[i]) > max_len:
                total_data[i] = total_data[i][:max_len]

        lab = [0] * len(total_data)

        self.x = torch.LongTensor(total_data)
        self.y = lab
        self.n_samples = len(total_data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples