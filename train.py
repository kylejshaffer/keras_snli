import argparse
import utils
import numpy as np
import os
import utils

from snli_model  import SNLIModel

def train(args):
	base_path = '/home/kyshaffe/Documents/datasets/snli_1.0'
	train_file = os.path.join(base_path, 'train.txt')
	dev_file = os.path.join(base_path, 'dev.txt')
	vocab_file = 'vocab.txt'

	with open(vocab_file, encoding='utf8', mode='r') as infile:
		vocab_size = len(infile.readlines())

	nn_object = SNLIModel(args, train_file, dev_file, vocab_size)
	nn_object.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, required=False, default=549364)
    parser.add_argument('--num_val_examples', type=int, required=False, default=9842)
    parser.add_argument('--print_freq', type=int, required=False, default=4194304)
    parser.add_argument('--seq_len', type=int, required=False, default=50)
    parser.add_argument('--recurrent_cell', type=str, required=False, default='GRU')
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=2)
    parser.add_argument('--embedding_dropout_rate', type=float, required=False, default=0.2)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--hidden_dim', type=int, required=False, default=512)
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--prev_model_path', type=str, required=False, default=None)
    args = parser.parse_args()

    train(args)
    # test(args)
