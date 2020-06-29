from tensorflow.compat.v1.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant
import time
import numpy as np
import pydot
import graphviz

class SiameseModel:
    
    def __init__(self, word_embedding, data, use_cudnn_lstm=False, plot_model_architecture=True):
        self.hidden_units = 300
        self.embed_model = word_embedding
        self.input_dim = word_embedding.embed_dim
        self.vocab_size = data.vocab_size
        self.left = data.premise
        self.right = data.hypothesis
        self.max_len = data.max_len
        self.dense_units = 32
        self.name = '{}_glove{}_lstm{}_dense{}'.format(str(int(time.time())),
                                                        self.input_dim,self.hidden_units,self.dense_units)
        
        
        embedding_matrix = np.zeros((self.vocab_size, self.input_dim))
        for word, i in data.vocab:
            embedding_vector = self.embed_model.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embed = layers.Embedding(input_dim=self.vocab_size, output_dim=self.input_dim, 
                                 embeddings_initializer=Constant(embedding_matrix), 
                                 input_length=self.max_len, mask_zero=True, trainable=False)
        #embed.trainable=False

        if use_cudnn_lstm:
            lstm = layers.CuDNNLSTM(self.hidden_units, input_shape=(None, self.input_dim),unit_forget_bias=True,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer='l2', name='lstm_layer')
        else:
            lstm = layers.LSTM(self.hidden_units, input_shape=(None, self.input_dim), unit_forget_bias=True,
                               activation = 'relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer='l2', name='lstm_layer')
        left_input = Input(shape=(self.max_len), name='input_1')
        right_input = Input(shape=(self.max_len), name='input_2')        

        embed_left = embed(left_input)
        embed_right = embed(right_input)

        print('embed:',embed_right.shape)

        left_output = lstm(embed_left)
        right_output = lstm(embed_right)
        print('lstm:',right_output.shape)
        l1_norm = lambda x: 1 - K.abs(x[0]-x[1])
        merged = layers.Lambda(function=l1_norm, output_shape=lambda x: x[0],name='L1_distance')([left_output, right_output])
        #merged = layers.concatenate([left_output, right_output])
        #lstm_2 = layers.LSTM(hidden_units, unit_forget_bias=True,
        #                      activation = 'relu', kernel_regularizer='l2', name='lstm_layer2' )(merged)
        print('merged:', merged.shape)
        dense_1 = layers.Dense(self.dense_units, activation='relu')(merged)
        print('dense1:', dense_1.shape)
        output = layers.Dense(3, activation='softmax', name='output_layer')(dense_1)
        print('output:',output.shape)
        self.model = Model(inputs=[left_input, right_input], outputs=output)

        self.compile()
                
        #if plot_model_architecture:
    #        plot_model(self.model, show_shapes=True, to_file=self.name+'.png')
        
    def compile(self):
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer, metrics=['accuracy'])
        
    def fit(self, target, validation_split=0.3, epochs=5, batch_size=128, patience=2):
        left_data = self.left
        right_data = self.right
        early_stopping = EarlyStopping(patience=patience, monitor='val_loss')
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1,  
                                            factor=0.5,
                                            min_lr=0.00001)
        callbacks = [early_stopping, learning_rate_reduction]
        
        self.history = self.model.fit([left_data, right_data], target, 
                                 validation_split=validation_split,
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks)        
        
    def predict(self, left_data, right_data):
        return self.model.predict([left_data, right_data])
    
    def evaluate(self, left_data, right_data, target, batch_size=128):
        return self.model.evaluate([left_data, right_data], target, batch_size=batch_size)

    def save_pretrained_weights(self, path='./model/pretrained_weights.h5'):
        self.model.save_weights(path)
        print('Save pretrained weights at location: ', path)
        
    def load_pretrained_weights(self, path='./model/pretrained_weights.h5'):
        self.model.load_weights(path, by_name=True, skip_mismatch=True)
        print('Loaded pretrained weights')
        self.compile()
        
    def save(self, model_folder=None):
        print('Saving model in SavedModel format ...')    
        if model_folder==None or not os.path.isdir(model_folder):
            model_folder = self.name
        os.mkdir(model_folder)
        self.model.save(model_folder)
        print('Saved model to disk')
        
    def load_activation_model(self):
        self.activation_model = Model(inputs=self.model.input[0], 
                                      outputs=self.model.get_layer('lstm_layer').output)
        
    def load(self, model_folder='./model/'):
        #use for encoder decoder alontg with load_activation
        # load json and create model
        json_file = open(model_folder + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_folder + 'model.h5')
        print('Loaded model from disk')
        
        self.model = loaded_model
        # loaded model should be compiled
        self.compile()
        self.load_activation_model()

    def visualize_metrics(self):
        epochs = len(self.history.history['accuracy'])
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        t = f.suptitle('Performance', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)
        epoch_list = list(range(1,epochs+1))
        ax1.plot(epoch_list, self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(epoch_list, self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(0, epochs+1, 5))
        ax1.set_ylabel('Accuracy Value')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Accuracy')
        l1 = ax1.legend(loc="best")
        
        ax2.plot(epoch_list, self.history.history['loss'], label='Train Loss')
        ax2.plot(epoch_list, self.history.history['val_loss'], label='Validation Loss')
        ax2.set_xticks(np.arange(0, epochs+1, 5))
        ax2.set_ylabel('Loss Value')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss')
        l2 = ax2.legend(loc="best")
        plt.show()



