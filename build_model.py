import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model
from keras.layers import LSTM,Bidirectional,Dense,TimeDistributed,Embedding,Input,Dropout,Lambda

sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("data/elmo_3", trainable=False)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
batch_size = 2
maxlen=25
idx2tag = {0: 'B-PROD', 1: 'I-PROD', 2: 'B-BRAND', 3: 'D-LESS', 4: 'B-QUAN', 5: 'B-DISC', 6: 'P-MORE',
           7: 'I-BRAND', 8: 'B-PRICE', 9: 'D-MORE', 10: 'P-LESS', 11: 'P-PRICE', 12: 'O'}
n_tags = len(idx2tag)

def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[maxlen])
                     },
                      signature="tokens",
                      as_dict=True)["elmo"]

def load_model():
  input_text = Input(shape=(maxlen,),dtype=tf.string)
  embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
  x = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1,dropout=0.2))(embedding)
  out = TimeDistributed(Dense(n_tags,activation='softmax'))(x)
  model = Model(input_text,out)
  model.load_weights("data/model_2.h5")
  return model

if "__main__" == __name__:
  build_model()