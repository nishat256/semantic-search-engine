import re
import sys
import numpy as np
from build_model import load_model
from semantic_search import fetch_top_n_records_from_db, apply_filter

idx2tag = {0: 'B-PROD', 1: 'I-PROD', 2: 'B-BRAND', 3: 'D-LESS', 4: 'B-QUAN', 5: 'B-DISC', 6: 'P-MORE',
           7: 'I-BRAND', 8: 'B-PRICE', 9: 'D-MORE', 10: 'P-LESS', 11: 'P-PRICE', 12: 'O'}

def make_prediction(query):
  maxlen = 25
  new_X = []
  X = [query.split(),"hello"]
  for seq in X:
    new_seq = []
    for i in range(maxlen):
      try:
        new_seq.append(seq[i])
      except:
        new_seq.append("PADword")
    new_X.append(new_seq)
  new_model = load_model()
  pred = new_model.predict(np.array(new_X))[0]
  pred = np.argmax(pred,axis=-1)
  pred_tags = []
  for i in pred:
    pred_tags.append(idx2tag[i])
  return pred_tags

def name_entity_recognition(query):
    if len(query.split()) <=2 :
      return {"prod_desc": query,
              "price_less":None,
              "price_more":None,
              "quantity":None,
              "price":None,
              "discount":None,
              "discount_less":None,
              "discount_more":None,
              "brand":None}
    prediction = make_prediction(query)
    prod_desc = ""
    price = None
    discount = None
    price_less =None
    price_more = None
    discount_more = None
    discount_less =None
    brand = None
    query_tokens = query.split()
    quantity = None
    for index,item in enumerate(prediction):
        if item in ['B-PROD','I-PROD']:
            if item == 'B-PROD':
                prod_desc += query_tokens[index]
            else:
                prod_desc += " "+query_tokens[index]
        elif item == 'B-QUAN':
            quantity = int(re.findall(r'[0-9]+',query_tokens[index])[0])
        elif item == 'P-LESS':
            price_less = True
        elif item == 'P-MORE':
            price_more = True
        elif item == 'B-PRICE':
            price = int(query_tokens[index])
        elif item == 'B-DISC':
            discount = int(query_tokens[index][:-1])
        elif item == 'D-LESS':
            discount_less = True
        elif item == 'D-MORE':
            discount_more = True
        elif item == 'B-BRAND':
            brand = query_tokens[index]
            
    if brand:
        prod_desc = brand +" "+prod_desc
    return {"prod_desc": prod_desc,
            "price_less":price_less,
            "price_more":price_more,
            "quantity":quantity,
            "price":price,
            "discount":discount,
            "discount_less":discount_less,
            "discount_more":discount_more,
            "brand":brand}

def main(query):
  ner_response = name_entity_recognition(query)
  response = fetch_top_n_records_from_db(ner_response['prod_desc'])
  response = response[['Product Description','Brand Name','Grammage','Discount','MRP','Special Offer']]
  response = apply_filter(response.dropna(),ner_response)
  if response.empty:
    return False
  return response.T.to_dict().values()

if __name__ == "__main__":
  prediction = list(main(sys.argv[1]))
  with open('result.txt','w') as fo:
    fo.write(str(prediction))