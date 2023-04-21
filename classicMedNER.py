#!/usr/bin/env python
# coding: utf-8

"""
Created on Sat April 22 03:20:31 2023
@author: ClassicCollins
"""
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize,sent_tokenize
#import sklearn
#from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional
import tensorflow as tf
from keras.models import load_model
import streamlit as st
from PIL import Image

# ### Loading the Dataset and IOB Formatting
model = load_model('model.h5')

def read_text_message(text_message):
    texts = str(text_message)
    return (texts)

def extract_xml(xml_doc):
    tree = ET.parse(xml_doc)
    root = tree.getroot()
    text = root.find('TEXT').text
    tags = root.findall('TAGS/*')

    # Create IOB format
    iob_tags = []
    for word in word_tokenize(text):
        # Initialize tag as outside entity (O)
        tag = 'O'
        # Check if word matches any tags
        for t in tags:
             if int(t.attrib['start']) <= text.find(word) < int(t.attrib['end']):
                # Identify tag type
                tag_type = t.attrib['TYPE']
                # Identify tag class (B)
                if tag == 'O':
                    tag = 'B-' + tag_type
        iob_tags.append(tag)
    #Identify tag class (I)    
    for j in range(len(iob_tags)):
        if iob_tags[j] == iob_tags[j-1] and iob_tags[j] != 'O':
            iob_tags[j]= "I-"+iob_tags[j].lower()
            

    #Print IOB format
    word_data = []
    tag_data = []
    sentence = []
    count = 1
    #Print words and tags
    for word, tag in zip(word_tokenize(text), iob_tags):
        word_data.append(word)
        tag_data.append(tag)
    #Print sentence numbers
    for text in word_data:
        sentence.append(f"Sentence: {str(count)}")
        if text == ".":
          count += 1
    #store in DataFrames
    df = pd.DataFrame([sentence,word_data, tag_data]).T
    df.columns = ['sentence_no',"words", "tags"]
    return df


a = extract_xml(xml_doc='file1.xml')
df1 = pd.DataFrame(a)
b = extract_xml(xml_doc = 'file2.xml')
df2 = pd.DataFrame(b)
c = extract_xml(xml_doc = 'file3.xml')
df3 = pd.DataFrame(c)

data = pd.concat([df3,df2,df1], ignore_index=True)
agg_func = lambda s:[(a, b) for a,b in zip(s["words"].values.tolist(),s['tags'].values.tolist())]
agg_data=data.groupby(['sentence_no']).apply(agg_func).reset_index().rename(columns={0:'word_tag_pair'})
agg_data['sentence']=agg_data['word_tag_pair'].apply(lambda sentence:" ".join([s[0] for s in sentence]))
agg_data['tags']=agg_data['word_tag_pair'].apply(lambda sentence:" ".join([s[1] for s in sentence]))
agg_data['tokenised_sentences']=agg_data['sentence'].apply(lambda x:x.split())
agg_data['tag_list']=agg_data['tags'].apply(lambda x:x.split())
agg_data['len_sentence']=agg_data['tokenised_sentences'].apply(lambda x:len(x))
agg_data['len_tag']=agg_data['tag_list'].apply(lambda x:len(x))
#agg_data['is_equal']=agg_data.apply(lambda row:1 if row['len_sentence']==row['len_tag'] else 0,axis=1)
#agg_data['is_equal'].value_counts()

sentences_list=agg_data['tokenised_sentences'].tolist()
tags_list=agg_data['tag_list'].tolist()

tokeniser= tf.keras.preprocessing.text.Tokenizer(lower=False,filters='')
tokeniser.fit_on_texts(sentences_list)


#print("Vocabulary size of Tokeniser ",len(tokeniser.word_index)+1) # Adding one since 0 is reserved for padding
#tokeniser.index_word[15]
encoded_sentence=tokeniser.texts_to_sequences(sentences_list)
tags=list(set(data["tags"].values))
num_tags=len(tags)
tags_map={tag:i for i,tag in enumerate(tags)}
reverse_tag_map={v: k for k, v in tags_map.items()}
encoded_tags=[[tags_map[w] for w in tag] for tag in tags_list]

max_sentence_length=max([len(s) for s in sentences_list])
max_len = max_sentence_length
#tags_map
padded_encoded_sentences = pad_sequences(maxlen=max_len,sequences=encoded_sentence,padding="post",value=0)
padded_encoded_tags=pad_sequences(maxlen=max_len,sequences=encoded_tags,padding="post",value=0)

target= [to_categorical(i,num_classes = num_tags) for i in  padded_encoded_tags]
#print("Shape of Labels  after converting to Categorical for first sentence: ",target[0].shape)
# ### Splitting The Data

X_test = padded_encoded_sentences
y_test = target

y_pred=model.predict(X_test) ## Predict using collins_medicalNER model on Test Data
# In[46]:
def collins_Predictionevaluation(test_data,preds,actual_preds):
    #print(actual_preds)
    #print("Shape of Test Data Array",test_data.shape)
    #print("Shape of Test Data Array",preds)
    y_actual=np.argmax(np.array(actual_preds),axis=2)
    y_pred=np.argmax(preds,axis=2)
    num_test_data=test_data.shape[0]
    print("Number of Test Data Points: ",num_test_data)
    data=pd.DataFrame()
    df_list=[]
    for i in range(num_test_data):
        test_str=list(test_data[i])
        df=pd.DataFrame()
        df['test_tokens']=test_str
        df['tokens']=df['test_tokens'].apply(lambda x:tokeniser.index_word[x] if x!=0 else '<PAD>')
        df['actual_target_index']=list(y_actual[i])
        df['pred_target_index']=list(y_pred[i])
        df['actual_target_tag']=df['actual_target_index'].apply(lambda x:reverse_tag_map[x])
        df['pred_target_tag']=df['pred_target_index'].apply(lambda x:reverse_tag_map[x])
        df['id']=i+1
        df_list.append(df)
    data=pd.concat(df_list)
    pred_data=data[data['tokens']!='<PAD>']
    accuracy=pred_data[pred_data['actual_target_tag']==pred_data['pred_target_tag']].shape[0]/pred_data.shape[0]
    
    
    return (pred_data ,accuracy)
pred_data,accuracy = collins_Predictionevaluation(X_test,y_pred,y_test)
e = pred_data['tokens'].tolist()
m = pred_data['actual_target_tag'].tolist()
z = zip(e,m)
y = list(z)
emzy = y
#print(emzy)

def main():
    st.image("image.jpg", caption='Promote “Unity In Human” Through Data, Technologies And Product Innovations', width=600, use_column_width="always")
    # Caption
    st.write("""
    ### [Home](https://parallelscore.com/ "Click to visit ParallelScore Website")
    ### st.title("Medical NER APP Built By Collins")"""
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Named Entity Recognition App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message_input = st.text_input("App will be ready Soon. See Sample text Below","Type/Paste Here")
    
    text_result = ""
    if st.button("Predict"):
        text_result = read_text_message(message_input)
    else:
        text_result = read_text_message("")
    st.success('SAMPLE DATA (FILE 1, 2, 3) OUTPUT: {}'.format(emzy))
    if st.button("About"):
        st.text("Medical Named Entity Recognition App. using Streamlite, Tensorflow.keras and Python. Built By Classic Collins. Contact:08037953669")
        st.text("Acknoledgement: Thanks to Streamlit for this platform and ParallellScore for providing the dataset.")

if __name__=='__main__':
    main()

