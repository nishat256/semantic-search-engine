import numpy as np 
import pandas as pd
import tensorflow as tf
from build_model import elmo_model, sess
from sklearn.metrics.pairwise import cosine_similarity

df_data = pd.read_csv('data/processed_data.csv')
vectors = df_data[[str(i) for i in range(1024)]]
vectors_array = vectors.values

def sentence_embedding(x):
  embeddings = elmo_model(
     x,
    signature="default",
    as_dict=True)["elmo"]
  response = sess.run(tf.math.reduce_sum(embeddings,1))
  return response


def fetch_top_n_records_from_db(query, num_records=20):
	query_vector = sentence_embedding([query])
	similarity = cosine_similarity(query_vector, vectors_array)
	top_n_indices = np.argpartition(similarity[0], -num_records)[-num_records:]
	result = df_data.loc[top_n_indices] 
	return result


def apply_filter(response,ner_response):
    price = ner_response['price']
    discount = ner_response['discount']
    quantity = ner_response['quantity']
    if not price and not discount and not quantity:
        return response
    if price:
        if ner_response['price_more']:
            response = response[response['MRP']>=price]
        else:
            response = response[response['MRP']<=price]
    if discount:
        if ner_response['discount_less']:
            response = response[response['Discount']<=(discount/float(100))]
        else:
            response = response[response['Discount']>=(discount/float(100))]
    if quantity:
        print(str(quantity))
        response = response[response['Grammage'].str.contains(str(quantity))]
    return response
