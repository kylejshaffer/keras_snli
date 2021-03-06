import os
import sys
import string

import numpy as np
import tensorflow as tf

from keras.utils import to_categorical

def get_bpe_vocab(vocab_file):
    special_toks = ['<s>', '</s>', '<UNK>', '<PAD>']
    c = {}
    word_idx = 1
    with tf.gfile.GFile(vocab_file, 'r') as infile:
        for line in infile:
            if line.strip() in special_toks:
                continue
            c[line.strip()] = word_idx
            word_idx += 1
        for st in special_toks:
            if st not in c.keys():
                if st == '<PAD>':
                    c[st] = 0
                else:
                    c[st] = max(c.values()) + 1
    print('VOCAB SIZE = {}'.format(len(c)))
    return c

def get_char_vocab(vocab_file):
    special_toks = ['<s>', '</s>', '<PAD>']
    char_map = {}
    word_idx = 1
    with tf.gfile.GFile(vocab_file, 'r') as infile:
        for line in infile:
            if line.replace('\n', '') in special_toks:
                continue
            char_map[line.replace('\n', '')] = word_idx
            word_idx += 1
        for st in special_toks:
            if st not in char_map.keys():
                if st == '<PAD>':
                    char_map[st] = 0
                else:
                    char_map[st] = max(char_map.values()) + 1
    print('CHARACTER VOCAB SIZE = {}'.format(len(char_map)))
    return char_map

def test():
    import matplotlib.pyplot as plt
    import string
    filter_chars = set(string.printable)

    word_vocab_file = 'C:/Users/kyshaffe/Documents/neural_lm/data/bpe/freq3-vocab25000/dic.txt'
    data_file = 'C:/Users/kyshaffe/Documents/neural_lm/data/bpe/freq3-vocab25000/valid.txt'
    vocab_file = 'C:/Users/kyshaffe/Documents/neural_lm/data/bpe/freq3-vocab25000/char_vocab.txt'

    word_vocab = get_bpe_vocab(word_vocab_file)
    char_vocab = get_char_vocab(vocab_file)

    data_obj = CharacterLMData(data_file, word_vocab, char_vocab, 40, 32)

    return data_obj

class LanguageModelData(object):
    def __init__(self, data_file, vocab, max_seq_len, batch_size):
        self.data_file = data_file
        self.vocab = vocab
        if self.vocab is not None:
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def get_tokid(self, word):
        if word in self.vocab.keys():
            tokid = self.vocab[word]
        else:
            tokid = self.vocab['<UNK>']
        return tokid

    def get_orig_word(self, tokid):
        orig_word = self.inv_vocab[int(tokid)]
        return orig_word

    def get_line(self):
        bos = '<s>'
        eos = '</s>'
        with tf.gfile.GFile(self.data_file, 'r') as infile:
            for line in infile:
                x_words = line.strip().split()
                y_words = x_words[:]
                # Truncate sentences that go past `max_seq_len`
                if len(x_words) > self.max_seq_len:
                    x_words = x_words[: self.max_seq_len]
                    y_words = y_words[: self.max_seq_len]
                # Insert special BOS/EOS tokens
                x_words.insert(0, bos)
                y_words.append(eos)
                # Lookup IDs
                x_toks = [self.get_tokid(w) for w in x_words]
                y_toks = [self.get_tokid(w) for w in y_words]
                yield x_toks, y_toks

    def padding(self, x_batch, y_batch, seq_lengths):
        if self.vocab is not None:
            pad = self.vocab['<PAD>']
        else:
            pad = 0
        max_length = max(seq_lengths)

        for idx, batch in enumerate(x_batch):
            if len(batch) < max_length:
                diff = max_length - len(batch)
                padding = [pad] * diff
                # Pad `x` sequence
                x_padded = batch + padding
                x_batch[idx] = x_padded
                # Pad `y` sequence
                y_padded = y_batch[idx] + padding
                y_batch[idx] = y_padded
        return x_batch, y_batch

    def generate_batches(self):
        while True:
            x_batch, y_batch, seq_lengths  = [], [], []
            for x_toks, y_toks in self.get_line():
                # Build up batches
                x_batch.append(x_toks)
                y_batch.append(y_toks)
                # Store lengths
                seq_lengths.append(len(x_toks))
                if len(x_batch) == self.batch_size:
                    x_batch_padded, y_batch_padded = self.padding(x_batch, y_batch, seq_lengths)
                    yield np.asarray(x_batch_padded), np.asarray(y_batch_padded)
                    # Reset batch containers
                    x_batch, y_batch, seq_lengths  = [], [], []
            if len(x_batch) > 0:
                x_batch_padded, y_batch_padded = self.padding(x_batch, y_batch, seq_lengths)
                yield np.asarray(x_batch_padded), np.asarray(y_batch_padded)

class SNLIData(LanguageModelData):
    def __init__(self, data_file, max_seq_len, batch_size):
        super(SNLIData, self).__init__(data_file, None, max_seq_len, batch_size)

    def get_line(self):
        with tf.gfile.GFile(self.data_file, 'r') as infile:
            for line in infile:
                left, right, label = line.strip().split('\t')
                left, right = left.strip().split(), right.strip().split()
                # if len(left) > self.max_seq_len:
                #     left = left[: self.max_seq_len]
                #     right = right[: self.max_seq_len]

                left = [int(i) for i in left]
                right = [int(i) for i in right]
                label = int(label)
                yield left, right, label

    def _padding(self, x_batch, y_batch, seq_lengths):
        if self.vocab is not None:
            pad = self.vocab['<PAD>']
        else:
            pad = 0
        max_length = max(seq_lengths)

        for idx, batch in enumerate(x_batch):
            if len(batch) < max_length:
                left_diff = max_length - len(batch)
                left_padding = [pad] * left_diff
                # Pad `x` sequence
                x_padded = batch + left_padding
                x_batch[idx] = x_padded
            if len(y_batch[idx]) < max_length:
                # Pad `y` sequence
                right_diff = max_length - len(y_batch[idx])
                right_padding = [pad] * right_diff
                y_padded = y_batch[idx] + right_padding
                y_batch[idx] = y_padded
        return x_batch, y_batch

    def generate_batches(self):
        while True:
            x_left_batch, x_right_batch, y_batch, seq_lengths = [], [], [], []
            for x_left_toks, x_right_toks, y in self.get_line():
                x_left_batch.append(x_left_toks)
                x_right_batch.append(x_right_toks)
                y_batch.append(y)
                seq_lengths.append(max(len(x_left_toks), len(x_right_toks)))
                if len(x_left_batch) == self.batch_size:
                    x_left_padded, x_right_padded = self._padding(x_left_batch, x_right_batch, seq_lengths)
                    # x_left_out = np.asarray(x_left_padded)
                    # x_right_out = np.asarray(x_right_padded)
                    # y_out = np.asarray(y_batch)
                    # print(x_left_out.shape, x_right_out.shape, y_out.shape)
                    yield [np.asarray(x_left_padded), np.asarray(x_right_padded)], to_categorical(np.asarray(y_batch))
                    
                    # Reset batch containers
                    x_left_batch, x_right_batch, y_batch, seq_lengths = [], [], [], []
            if len(x_left_batch) > 0:
                x_left_padded, x_right_padded = self._padding(x_left_batch, x_right_batch, seq_lengths)
                yield [np.asarray(x_left_padded), np.asarray(x_right_padded)], to_categorical(np.asarray(y_batch))

class BiRNNData(LanguageModelData):
    def __init__(self, data_file, vocab, max_seq_len, batch_size):
        super(BiRNNData, self).__init__(data_file, vocab, max_seq_len, batch_size)

    def padding(self, x_batch, y_batch, seq_lengths):
        pad = self.vocab['<PAD>']
        max_length = max(seq_lengths)
        x_batch_bwd, y_batch_bwd = [], []

        for idx, batch in enumerate(x_batch):
            if len(batch) < max_length:
                diff = max_length - len(batch)
                padding = [pad] * diff
                # Pad `x` sequence
                x_bwd = list(reversed(batch))
                x_padded = batch + padding
                x_batch[idx] = x_padded
                x_padded_bwd = x_bwd + padding
                x_batch_bwd.append(x_padded_bwd)
                # Pad `y` sequence
                y_bwd = list(reversed(y_batch[idx]))
                y_padded = y_batch[idx] + padding
                y_batch[idx] = y_padded
                y_padded_bwd = y_bwd + padding
                y_batch_bwd.append(y_padded_bwd)
            else:
                x_bwd = list(reversed(batch))
                y_bwd = list(reversed(y_batch[idx]))
                x_batch_bwd.append(x_bwd)
                y_batch_bwd.append(y_bwd)

        return x_batch, x_batch_bwd, y_batch, y_batch_bwd

    def generate_batches(self):
        while True:
            x_batch, y_batch, seq_lengths  = [], [], []
            for x_toks, y_toks in self.get_line():
                # Build up batches
                x_batch.append(x_toks)
                y_batch.append(y_toks)
                # Store lengths
                seq_lengths.append(len(x_toks))
                if len(x_batch) == self.batch_size:
                    x_batch_padded, x_batch_padded_bwd, y_batch_padded, y_batch_padded_bwd = self.padding(x_batch, y_batch, seq_lengths)
                    yield [np.asarray(x_batch_padded), np.asarray(x_batch_padded_bwd)], \
                          [np.asarray(y_batch_padded), np.asarray(y_batch_padded_bwd)]
                    # Reset batch containers
                    x_batch, y_batch, seq_lengths  = [], [], []
            if len(x_batch) > 0:
                x_batch_padded, x_batch_padded_bwd, y_batch_padded, y_batch_padded_bwd = self.padding(x_batch, y_batch, seq_lengths)
                yield [np.asarray(x_batch_padded), np.asarray(x_batch_padded_bwd)], \
                      [np.asarray(y_batch_padded), np.asarray(y_batch_padded_bwd)]

class CharacterLMData(LanguageModelData):
    def __init__(self, data_file, vocab, char_vocab, max_seq_len, batch_size):
        self.char_vocab = char_vocab
        self.filter_chars = set(string.printable)
        # Placeholder - longest word in training set
        self.max_word_length = 24
        super(CharacterLMData, self).__init__(data_file, vocab, max_seq_len, batch_size)

    def get_char_ids(self, word):
        char_ids = [self.char_vocab[ch] for ch in word]
        return char_ids

    def pad_chars(self, char_batch):
        pad = self.get_tokid('<PAD>')
        for idx, char_line in enumerate(char_batch):
            padded_char_line = []
            for word in char_line:
                if len(word) < self.max_word_length:
                    diff = self.max_word_length - len(word)
                    char_padding = [pad] * diff
                    padded_word = word + char_padding
                    padded_char_line.append(padded_word)
                else:
                    padded_char_line.append(word)
            char_batch[idx] = padded_char_line
        return char_batch

    def get_line(self):
        bos = '<s>'
        eos = '</s>'
        with tf.gfile.GFile(self.data_file, 'r') as infile:
            for line in infile:
                line = line.replace('”', "''").replace('“', '``').replace('ʼ', "'")
                line = ''.join(filter(lambda x: x in self.filter_chars, line.strip()))
                x_words = line.strip().split()
                y_words = x_words[:]
                # Truncate sentences that go past `max_seq_len`
                if len(x_words) > self.max_seq_len:
                    x_words = x_words[: self.max_seq_len]
                    y_words = y_words[: self.max_seq_len]
                # Insert special BOS/EOS tokens
                y_words.append(eos)
                # Lookup IDs
                x_char_toks = [self.get_char_ids(word) for word in x_words]
                x_char_toks.insert(0, [self.get_tokid(bos)])
                # x_toks = [self.get_tokid(w) for w in x_words]
                y_toks = [self.get_tokid(w) for w in y_words]
                yield x_char_toks, y_toks

    def padding(self, x_batch, y_batch, seq_lengths):
        pad = self.vocab['<PAD>']
        max_seq_length = max(seq_lengths)

        # Pad batch of input characters
        x_batch_padded = self.pad_chars(x_batch)
        for idx, batch in enumerate(y_batch):
            if len(batch) < max_seq_length:
                # Pad `y` sequence
                diff = max_seq_length - len(batch)
                word_padding = [pad] * diff
                y_padded = batch + word_padding
                y_batch[idx] = y_padded
                # Pad `x` character sequence
                x_pad_words = [[pad] * self.max_word_length for _ in range(diff)]
                x_batch_padded[idx].extend(x_pad_words)
        return x_batch_padded, y_batch

    def generate_batches(self):
        while True:
            x_batch, y_batch, seq_lengths  = [], [], []
            for x_toks, y_toks in self.get_line():
                # Build up batches
                x_batch.append(x_toks)
                y_batch.append(y_toks)
                # Store lengths
                seq_lengths.append(len(y_toks))
                if len(x_batch) == self.batch_size:
                    x_batch_padded, y_batch_padded = self.padding(x_batch, y_batch, seq_lengths)
                    yield np.asarray(x_batch_padded), np.asarray(y_batch_padded)
                    # Reset batch containers
                    x_batch, y_batch, seq_lengths  = [], [], []
            if len(x_batch) > 0:
                x_batch_padded, y_batch_padded = self.padding(x_batch, y_batch, seq_lengths)
                yield np.asarray(x_batch_padded), np.asarray(y_batch_padded)


if __name__ == '__main__':
    test()
