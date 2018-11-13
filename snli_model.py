import argparse
import keras
import keras.backend as K
import numpy as np
import utils

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Embedding, Input, GRU, LSTM, Lambda
from keras.layers.merge import Concatenate, Dot, Multiply, Subtract
from keras.layers.wrappers import Bidirectional


class SNLIModel(object):
    def __init__(self, args, train_file, dev_file, vocab_size):
        self.rec_cell_map = {'GRU': GRU, 'LSTM': LSTM}
        self.train_file = train_file
        self.dev_file = dev_file
        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.num_train_examples = args.num_train_examples
        self.num_val_examples = args.num_val_examples
        self.max_seq_len = args.seq_len
        self.embedding_dim = args.embedding_dim
        self.recurrent_cell = args.recurrent_cell
        self.hidden_dim = args.hidden_dim
        self.num_hidden_layers = args.num_hidden_layers
        self.opt_string = args.optimizer
        self.learning_rate = args.learning_rate
        self._choose_optimizer()
        self._build_layers()
        self.build_graph()

    def _choose_optimizer(self):
        assert self.opt_string in {'adagrad', 'adam', 'sgd', 'momentum', 'rmsprop'}, 'Please select valid optimizer!'

        learning_rate = self.learning_rate

        # Set optimizer
        if self.opt_string == 'adagrad':
            self.optimizer = keras.optimizers.Adagrad(lr=learning_rate)
        elif self.opt_string == 'adam':
            self.optimizer = keras.optimizers.Adam(lr=learning_rate)
        elif self.opt_string == 'rmsprop':
            self.optimizer = keras.optimizers.RMSprop(lr=learning_rate)
        elif self.opt_string == 'sgd':
            self.optimizer = keras.optimizers.SGD(lr=learning_rate)
        elif self.opt_string == 'momentum':
            self.optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.99)
        else:
            'Invalid optimizer selected - exiting'
            sys.exit(1)

    def _build_layers(self):
        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)
        rec_cell = self.rec_cell_map[self.recurrent_cell]
        self.dropout_layer = Dropout(0.2)
        self.recurrent_layers = [Bidirectional(rec_cell(units=self.hidden_dim, return_sequences=True, name='rec_{}'.format(_))) for _ in range(self.num_hidden_layers)]
        self.max_pool_layer = Lambda(lambda x: K.max(x, axis=1))
        self.dense_hidden = Dense(units=512, activation='relu', name='dense_hidden')
        self.dense_out = Dense(units=3, activation='softmax', name='dense_out')

    def build_graph(self):
        in_layer_left = Input((None, ), name='input_left')
        in_layer_right = Input((None, ), name='input_right')

        left_embedded = self.embedding_layer(in_layer_left)
        right_embedded = self.embedding_layer(in_layer_right)

        left_embedded = self.dropout_layer(left_embedded)
        right_embedded = self.dropout_layer(right_embedded)
        if len(self.recurrent_layers) > 1:
            for i, layer in enumerate(self.recurrent_layers[:-1]):
                left_embedded = layer(left_embedded)
                right_embedded = layer(right_embedded)
        final_recurrent_layer = self.recurrent_layers[-1]
        left_encoded = final_recurrent_layer(left_embedded)
        right_encoded = final_recurrent_layer(right_embedded)

        left_pooled = self.max_pool_layer(left_encoded)
        right_pooled = self.max_pool_layer(right_encoded)

        subtract_tensor = Subtract()([left_pooled, right_pooled])
        subtract_tensor = Lambda(lambda x: K.abs(x))(subtract_tensor)
        concat_tensor = Concatenate()([left_pooled, right_pooled])
        mult_tensor = Multiply()([left_pooled, right_pooled])
        # dot_tensor = Dot(axes=[-1, -1])([left_pooled, right_pooled])

        total_concat = Concatenate()([concat_tensor, mult_tensor, subtract_tensor])
        logits = self.dense_hidden(total_concat)
        logits_normed = self.dense_out(logits)

        y_ph = K.tf.placeholder(K.tf.int32, shape=(None, None), name='y_ph')

        model = Model(inputs=[in_layer_left, in_layer_right], outputs=logits_normed)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['acc'])
        model.summary()
        self.model = model

    def sparse_loss(self, y_true, y_pred, from_logits=True):
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits)

    def train(self):
        np.random.seed(7)
        n_train_iters = self.num_train_examples // self.batch_size
        n_valid_iters = self.num_val_examples // self.batch_size

        train_data = utils.SNLIData(data_file=self.train_file, max_seq_len=self.max_seq_len, batch_size=self.batch_size)
        dev_data = utils.SNLIData(data_file=self.dev_file, max_seq_len=self.max_seq_len, batch_size=self.batch_size)

        train_datagen = train_data.generate_batches()
        dev_datagen = dev_data.generate_batches()

        ckpt_fname = 'snli_model_1024dim_{epoch:02d}-{val_acc:.2f}.h5'
        ckpt = ModelCheckpoint(ckpt_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        ckpt_list = [ckpt]
        self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters,
                                epochs=self.epochs, validation_data=dev_datagen, validation_steps=n_valid_iters,
                                callbacks=ckpt_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, required=False, default=20000)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--recurrent_cell', type=str, required=False, default='GRU')
    parser.add_argument('--hidden_dim', type=int, required=False, default=512)
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=3)
    args = parser.parse_args()

    snli_model = SNLIModel(args)

