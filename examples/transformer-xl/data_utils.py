import os, sys
import glob
import pdb
from collections import Counter, OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vocabulary import Vocab
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class CSQAIterator(object):
    def __init__(self, data, bsz):
        """
            data: [encoded, labels]
            encoded: [QA1, QA2, QA3, QA4, QA5]
            QAx: [tensor]
        """

        self.bsz = bsz

        self.encoded_0 = data[0][0] # List
        self.encoded_1 = data[0][1]
        self.encoded_2 = data[0][2]
        self.encoded_3 = data[0][3]
        self.encoded_4 = data[0][4]
        
        self.labels = data[1] # Tensor

        self.n_step = self.labels.size(0) // bsz
        self.n_samples = self.labels.size(0)
        self.sequence_array = np.arange(self.n_samples)

    def get_batch(self, index_list):

        subencoded_0 = []
        subencoded_1 = []
        subencoded_2 = []
        subencoded_3 = []
        subencoded_4 = []
        mask_idx_0 = []
        mask_idx_1 = []
        mask_idx_2 = []
        mask_idx_3 = []
        mask_idx_4 = []
        sublabels = []

        for idx in index_list:
            subencoded_0.append(self.encoded_0[idx])
            subencoded_1.append(self.encoded_1[idx])
            subencoded_2.append(self.encoded_2[idx])
            subencoded_3.append(self.encoded_3[idx])
            subencoded_4.append(self.encoded_4[idx])
            sublabels.append(self.labels[idx])
            mask_idx_0.append(torch.ones(self.encoded_0[idx].shape[0]))
            mask_idx_1.append(torch.ones(self.encoded_1[idx].shape[0]))
            mask_idx_2.append(torch.ones(self.encoded_2[idx].shape[0]))
            mask_idx_3.append(torch.ones(self.encoded_3[idx].shape[0]))
            mask_idx_4.append(torch.ones(self.encoded_4[idx].shape[0]))
        
        subencoded_0 = pad_sequence(subencoded_0)
        subencoded_1 = pad_sequence(subencoded_1)
        subencoded_2 = pad_sequence(subencoded_2)
        subencoded_3 = pad_sequence(subencoded_3)
        subencoded_4 = pad_sequence(subencoded_4)
        atten_mask_0 = 1 - pad_sequence(mask_idx_0)
        atten_mask_1 = 1 - pad_sequence(mask_idx_1)
        atten_mask_2 = 1 - pad_sequence(mask_idx_2)
        atten_mask_3 = 1 - pad_sequence(mask_idx_3)
        atten_mask_4 = 1 - pad_sequence(mask_idx_4)
        sublabels = torch.LongTensor(sublabels)

        return subencoded_0, subencoded_1, subencoded_2, subencoded_3, subencoded_4, \
                atten_mask_0, atten_mask_1, atten_mask_2, atten_mask_3, atten_mask_4, sublabels

    def get_varlen_iter(self, start=0):
        sample_array = np.random.permutation(self.n_samples)
        for i in range(self.n_step):
            sub_index = sample_array[i*self.bsz:i*self.bsz+self.bsz]
            yield self.get_batch(sub_index)

    def get_fixlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        for i in range(self.n_step):
            sub_index = self.sequence_array[i*self.bsz:i*self.bsz+self.bsz]
            yield self.get_batch(sub_index)

    def __iter__(self):
        return self.get_fixlen_iter()


class SST2Iterator(object):
    def __init__(self, data, bsz):
        """
            data: [encoded, labels]
        """

        self.bsz = bsz

        self.encoded = data[0]
        self.labels = data[1] # Tensor

        self.n_step = self.labels.size(0) // bsz
        self.n_samples = self.labels.size(0)
        self.sequence_array = np.arange(self.n_samples)

    def get_batch(self, index_list):

        subencoded = []
        mask_idx = []
        sublabels = []

        for idx in index_list:
            subencoded.append(self.encoded[idx])
            sublabels.append(self.labels[idx])
            mask_idx.append(torch.ones(self.encoded[idx].shape[0]))
        
        subencoded = pad_sequence(subencoded)
        mask_idx = 1 - pad_sequence(mask_idx)

        import pdb; pdb.set_trace()
        





        # mask_idx = pad_sequence(mask_idx)
        sublabels = torch.LongTensor(sublabels)

        return subencoded, mask_idx, sublabels

    def get_varlen_iter(self, start=0):
        sample_array = np.random.permutation(self.n_samples)
        for i in range(self.n_step):
            sub_index = sample_array[i*self.bsz:i*self.bsz+self.bsz]
            yield self.get_batch(sub_index)

    def get_fixlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        for i in range(self.n_step):
            sub_index = self.sequence_array[i*self.bsz:i*self.bsz+self.bsz]
            yield self.get_batch(sub_index)

    def __iter__(self):
        return self.get_fixlen_iter()


class CSQADataset(Dataset):

    def __init__(self, data):
        super(CSQADataset, self).__init__()
        """
            data: [encoded, labels]
            encoded: [QA1, QA2, QA3, QA4, QA5]
            QAx: [tensor]
        """

        self.encoded = data[0]
        self.labels = data[1]

        self.n_question = self.labels.shape[0]

    def __len__(self):

        return self.n_question

    def __getitem__(self, index):

        return self.encoded[:, 0, index], self.encoded[:, 1, index], \
            self.encoded[:, 2, index], self.encoded[:, 3, index], \
            self.encoded[:, 4, index], self.labels[index]


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
        shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        elif self.dataset == 'csqa':
            self.vocab.count_csqa(os.path.join(path, 'train_rand_split.jsonl'), add_cls_token=True)
            self.vocab.count_csqa(os.path.join(path, 'dev_rand_split.jsonl'), add_cls_token=True)
            self.vocab.count_csqa(os.path.join(path, 'test_rand_split_no_answers.jsonl'), add_cls_token=True)

        elif self.dataset == 'sst2':
            self.vocab.count_sst2(os.path.join(path, 'train.tsv'), add_cls_token=True)
            self.vocab.count_sst2(os.path.join(path, 'dev.tsv'), add_cls_token=True)
            self.vocab.count_sst2(os.path.join(path, 'test.tsv'), add_cls_token=True)

        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)
        elif self.dataset == 'csqa':
            self.train = self.vocab.encode_csqa_file(
                os.path.join(path, 'train_rand_split.jsonl'), ordered=True, add_cls_token=True)
            self.valid = self.vocab.encode_csqa_file(
                os.path.join(path, 'dev_rand_split.jsonl'), ordered=True, add_cls_token=True)
        elif self.dataset == 'sst2':
            self.train = self.vocab.encode_sst2_file(
                os.path.join(path, 'train.tsv'), add_cls_token=True)
            self.valid = self.vocab.encode_sst2_file(
                os.path.join(path, 'dev.tsv'), add_cls_token=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
            elif self.dataset == 'csqa':
                data_iter = CSQAIterator(self.train, *args, **kwargs)
            elif self.dataset == 'sst2':
                data_iter = SST2Iterator(self.train, *args, **kwargs)

                # dataset = CSQADataset(self.train)
                # data_iter = DataLoader(dataset, *args, shuffle=True, 
                #                     num_workers=4, drop_last=False, pin_memory=True)

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)
            elif self.dataset == 'csqa':
                data_iter = CSQAIterator(self.valid, *args, **kwargs)
            elif self.dataset == 'sst2':
                data_iter = SST2Iterator(self.valid, *args, **kwargs)

                # dataset = CSQADataset(self.valid)
                # data_iter = DataLoader(dataset, *args, shuffle=False, 
                #                     num_workers=4, drop_last=False, pin_memory=True)
        return data_iter


def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, 'cache.pt')
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['csqa', 'sst2']:
            kwargs['special'] = ['<eos>']
        elif dataset in ['enwik8', 'text8']:
            pass

        corpus = Corpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
