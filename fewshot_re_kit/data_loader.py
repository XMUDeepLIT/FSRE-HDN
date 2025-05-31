import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import pdb

Relations = json.load(open('./data/pid2name.json', 'r'))

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, test=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.test =test

    def __getraw__(self, item):
        word, pos1, pos2 = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word, pos1, pos2

    def __additem__(self, d, word, pos1, pos2):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': []}
        query_set = {'word': [], 'pos1': [], 'pos2': []}
        relation_example = {'word': [], 'pos1': [], 'pos2': []}
        relation_example1 = {'word': [], 'pos1': [], 'pos2': []}
        query_label, support_label = [], []
        Relation_name = []
        Q_na = int(self.na_rate * self.Q)

        target_classes = random.sample(self.classes, self.N)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            label_temp = [0] * self.N
            label_temp[i] = 1
            for j in indices:
                word, pos1, pos2 = self.__getraw__(self.json_data[class_name][j])
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2)
                    query_label.append(label_temp)
                else:
                    self.__additem__(query_set, word, pos1, pos2)
                    support_label.append(label_temp)
                count += 1

        # NA
        label_temp = [0] * self.N
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(list(range(len(self.json_data[cur_class]))),1, False)[0]
            word, pos1, pos2 = self.__getraw__( self.json_data[cur_class][index])
            self.__additem__(query_set, word, pos1, pos2)
            query_label.append(label_temp)



        for i, class_name in enumerate(target_classes):
            for k in range(self.K):
                if(random.uniform(0,1)>0.4) or self.test:
                    r = " : ".join(Relations[class_name]) + " : "
                    r = self.encoder.tokenizer.encode(r)
                    example = support_set['word'][i * self.K + k]
                    pro = len(r[:-1]) - 1
                    p1 = support_set['pos1'][i * self.K + k] + pro
                    p2 = support_set['pos2'][i * self.K + k] + pro
                    r = r[:-1] + example[1:-1] + r[-1:]
                else:
                    p1 = support_set['pos1'][i * self.K + k]
                    p2 = support_set['pos2'][i * self.K + k]
                    r = support_set['word'][i * self.K + k]

                self.__additem__(relation_example, r, p1, p2)


        for i, class_name in enumerate(target_classes):
            for k in range(self.K):
                if (random.uniform(0, 1) > 0.4) or self.test:
                    r = " : ".join(Relations[class_name]) + " : "
                    r = self.encoder.tokenizer.encode(r)
                    example = support_set['word'][i * self.K + k]
                    pro = len(r[:-1]) - 1
                    p1 = support_set['pos1'][i * self.K + k] + pro
                    p2 = support_set['pos2'][i * self.K + k] + pro
                    r = r[:-1] + example[1:-1] + r[-1:]
                else:
                    p1 = support_set['pos1'][i * self.K + k]
                    p2 = support_set['pos2'][i * self.K + k]
                    r = support_set['word'][i * self.K + k]

                self.__additem__(relation_example1, r, p1, p2)

        relation = []
        for i, class_name in enumerate(target_classes):
            r = " : ".join(Relations[class_name])
            r = self.encoder.tokenizer.encode(r)
            # r = self.encoder.tokenizer.decode(r)
            if (random.uniform(0, 1) > 2.0) and (not self.test):
                example = support_set['word'][i * self.K + 0]
                r = r[:-1] + example[1:-1] + r[-1:]

            relation.append(r)

        if(self.test):
            labels = support_label + query_label
        else:
            labels = support_label + query_label

        return query_set, relation, labels, relation_example, relation_example1
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_samples = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_relation = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_relation_examples = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_relation_examples1 = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []

    samples, relation, labels, relation_examples,relation_examples1 = zip(*data)
    for i in range(len(samples)):
        for k in ['word', 'pos1', 'pos2']:
            batch_samples[k] += samples[i][k]
            batch_relation_examples[k] += relation_examples[i][k]
            batch_relation_examples1[k] += relation_examples1[i][k]

        batch_label += labels[i]
        batch_relation['word'] += relation[i]


    Max_word = max([len(x) for x in batch_samples['word']])
    Max_rel = max([len(x) for x in batch_relation['word']])
    Max_rel_example = max([len(x) for x in batch_relation_examples['word']])
    Max_rel_example1 = max([len(x) for x in batch_relation_examples1['word']])

    batch_samples['mask'] = [[1.0] * len(x) + [0.0] * (Max_word - len(x)) for x in batch_samples['word']]
    batch_samples['word'] = [x + [0] * (Max_word - len(x)) for x in batch_samples['word']]

    batch_relation['mask'] = [[1.0] * len(x) + [0.0] * (Max_rel - len(x)) for x in batch_relation['word']]
    batch_relation['word'] = [x + [0] * (Max_rel - len(x)) for x in batch_relation['word']]

    batch_relation_examples['mask'] = [[1.0] * len(x) + [0.0] * (Max_rel_example - len(x)) for x in batch_relation_examples['word']]
    batch_relation_examples['word'] = [x + [0] * (Max_rel_example - len(x)) for x in batch_relation_examples['word']]

    batch_relation_examples1['mask'] = [[1.0] * len(x) + [0.0] * (Max_rel_example1 - len(x)) for x in batch_relation_examples1['word']]
    batch_relation_examples1['word'] = [x + [0] * (Max_rel_example1 - len(x)) for x in batch_relation_examples1['word']]

    for k in ['word', 'pos1', 'pos2', 'mask']:
        batch_samples[k] = torch.tensor(batch_samples[k]).long()
        batch_relation[k] = torch.tensor(batch_relation[k]).long()
        batch_relation_examples[k] = torch.tensor(batch_relation_examples[k]).long()
        batch_relation_examples1[k] = torch.tensor(batch_relation_examples1[k]).long()

    batch_label = torch.tensor(batch_label).long()
    return batch_samples, batch_relation, batch_relation_examples, batch_relation_examples1, batch_label

def get_loader(name, encoder, N, K, Q, batch_size, num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data', test =False):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, test=test)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            # num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word  = self.__getraw__(
                        self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(list(range(len(self.json_data[cur_class]))),1, False)[0]
            word = self.__getraw__(
                    self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


