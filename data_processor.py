import torch
import pandas as pd
import json
from random import sample


def load_json(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


class CustomDataset():
    def __init__(self, domain, sample_size, train=True):
        self.x, self.y = self.data_split(domain, sample_size, train)
        self.n_samples = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples



    @staticmethod
    def data_split(domain, train):
        human_data = load_json("data/set1_human.json")
        machine_data = load_json("data/set1_machine.json")
        human_data2 = load_json("data/set2_human.json")
        machine_data2 = load_json("data/set2_machine.json")

        max_hm1_prompt = max([len(i['prompt']) for i in human_data])
        max_hm1_txt = max([len(i['txt']) for i in human_data])
        max_mc1_prompt = max([len(i['prompt']) for i in machine_data])
        max_mc1_txt = max([len(i['txt']) for i in machine_data])
        max_hm2_prompt = max([len(i['prompt']) for i in human_data2])
        max_hm2_txt = max([len(i['txt']) for i in human_data2])
        max_mc2_prompt = max([len(i['prompt']) for i in machine_data2])
        max_mc2_txt = max([len(i['txt']) for i in machine_data2])

        max_domain1_prompt = max(max_hm1_prompt, max_mc1_prompt)
        max_domain1_text = 2019
        max_domain2_prompt = max(max_hm2_prompt, max_mc2_prompt)
        max_domain2_text = 1496

        # padding
        for dt in human_data:
            if len(dt['prompt']) < max_domain1_prompt:
                dt['prompt'] += [0] * (max_domain1_prompt - len(dt['prompt']))
            elif len(dt['prompt']) > max_domain1_prompt:
                dt['prompt'] = dt['prompt'][:max_domain1_prompt]
            if len(dt['txt']) < max_domain1_text:
                dt['txt'] += [0] * (max_domain1_text - len(dt['txt']))
            elif len(dt['txt']) > max_domain1_text:
                dt['txt'] = dt['txt'][:max_domain1_text]

        for dt in machine_data:
            if len(dt['prompt']) < max_domain1_prompt:
                dt['prompt'] += [0] * (max_domain1_prompt - len(dt['prompt']))
            elif len(dt['prompt']) > max_domain1_prompt:
                dt['prompt'] = dt['prompt'][:max_domain1_prompt]
            if len(dt['txt']) < max_domain1_text:
                dt['txt'] += [0] * (max_domain1_text - len(dt['txt']))
            elif len(dt['txt']) > max_domain1_text:
                dt['txt'] = dt['txt'][:max_domain1_text]

        for dt in human_data2:
            if len(dt['prompt']) < max_domain2_prompt:
                dt['prompt'] += [0] * (max_domain2_prompt - len(dt['prompt']))
            elif len(dt['txt']) > max_domain2_prompt:
                dt['prompt'] = dt['prompt'][:max_domain2_prompt]

            if len(dt['txt']) < max_domain2_text:
                dt['txt'] += [0] * (max_domain2_text - len(dt['txt']))
            elif len(dt['txt']) > max_domain2_text:
                dt['txt'] = dt['txt'][:max_domain2_text]

        for dt in machine_data2:
            if len(dt['prompt']) < max_domain2_prompt:
                dt['prompt'] += [0] * (max_domain2_prompt - len(dt['prompt']))
            elif len(dt['prompt']) > max_domain2_prompt:
                dt['prompt'] = dt['prompt'][:max_domain2_prompt]

            if len(dt['txt']) < max_domain2_text:
                dt['txt'] += [0] * (max_domain2_text - len(dt['txt']))
            elif len(dt['txt']) > max_domain2_text:
                dt['txt'] = dt['txt'][:max_domain2_text]

        hm_domain1 = [h['prompt'] + [1] + h['txt'] for h in human_data]
        mc_domain1 = [m['prompt'] + [1] + m['txt'] for m in machine_data]
        hm_domain2 = [h['prompt'] + [1] + h['txt'] for h in human_data2]
        mc_domain2_id0 = [m['prompt'] + [1] + m['txt'] for m in machine_data2]

        # random sampling data from each parts
        total_data = []
        HM1_SIZE, MC1_SIZE, MH2_SIZE, MC2_SIZE = sample_size[0], sample_size[1], sample_size[2], sample_size[3]
        if train:
            if domain == 1:  # domain 1 data
                total_data += (sample(hm_domain1, HM1_SIZE) + mc_domain1)
            elif domain == 2:  # domain 2 data
                total_data += hm_domain2 + sample(mc_domain2_id0, MC2_SIZE)
        else:  # test data
            if domain == 1:
                total_data += (sample(hm_domain1, 500) + sample(mc_domain1, 500))
            elif domain == 2:
                total_data += (sample(hm_domain2, 100) + sample(mc_domain2_id0, 100))

        labels = []
        if train:
            if domain == 1:
                labels += [1] * 3500 + [0] * 3500
            else:
                labels += [1] * 100 + [0] * 100
        else:
            if domain == 1:
                labels += [1] * 500 + [0] * 500
            else:
                labels += [1] * 100 + [0] * 100

        return torch.LongTensor(total_data), labels


class TestSet():
    def __init__(self, data_path):
        data = load_json(data_path)

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


def word_distribution(path, file_name, min_frequency=3000):
    current = json.load(open(path, encoding='utf-8'))
    all_words = []
    for d in current:
        all_words += d['prompt'] + d['txt']

    counts = []
    for i in range(5000):
        counts.append(all_words.count(i))

    min_freqs = nsmallest(min_frequency, counts)
    min_freq_words = []
    curr_freq = None
    for i in range(min_frequency):
        if curr_freq is None or min_freqs[i] != curr_freq:
            curr_freq = min_freqs[i]
            indices = [j for j, k in enumerate(counts) if k == min_freqs[i]]
            min_freq_words += indices
        # min_freq_words[i] = counts.index(min_freq_words[i])


    for c in current:
        for p in range(len(c['prompt'])):
            if c['prompt'][p] in min_freq_words:
                c['prompt'][p] = -1
        c['prompt'] = list(filter((-1).__ne__, c['prompt']))

        for t in range(len(c['txt'])):
            if c['txt'][t] in min_freq_words:
                c['txt'][t] = -1
        c['txt'] = list(filter((-1).__ne__, c['txt']))

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(current, f, ensure_ascii=False, indent=4)

    return min_freq_words