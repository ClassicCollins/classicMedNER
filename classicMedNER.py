#!/usr/bin/env python
# coding: utf-8

try: 
    import livelossplot
except:
    get_ipython().system('pip install livelossplot')
    import livelossplot
from livelossplot.tf_keras import PlotLossesCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# ### Loading the Dataset and IOB Formatting

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


data.head(10)



data.sentence_no.value_counts()


data.shape

data.tags.unique()


data.isnull().sum()

data.info()

data['tags'].value_counts() #Not balanced

len(data['tags'].value_counts())

agg_func = lambda s:[(a, b) for a,b in zip(s["words"].values.tolist(),s['tags'].values.tolist())]


agg_data=data.groupby(['sentence_no']).apply(agg_func).reset_index().rename(columns={0:'word_tag_pair'})
agg_data.head()


agg_data['sentence']=agg_data['word_tag_pair'].apply(lambda sentence:" ".join([s[0] for s in sentence]))
agg_data['tags']=agg_data['word_tag_pair'].apply(lambda sentence:" ".join([s[1] for s in sentence]))


agg_data.shape

agg_data.head()


agg_data['tokenised_sentences']=agg_data['sentence'].apply(lambda x:x.split())
agg_data['tag_list']=agg_data['tags'].apply(lambda x:x.split())
agg_data.head()

agg_data['len_sentence']=agg_data['tokenised_sentences'].apply(lambda x:len(x))
agg_data['len_tag']=agg_data['tag_list'].apply(lambda x:len(x))
agg_data['is_equal']=agg_data.apply(lambda row:1 if row['len_sentence']==row['len_tag'] else 0,axis=1)
agg_data['is_equal'].value_counts()

agg_data.shape

sentences_list=agg_data['tokenised_sentences'].tolist()
tags_list=agg_data['tag_list'].tolist()

print("Number of Sentences in the Data: ",len(sentences_list))
print("Are number of Sentences and Tag list equal: ",len(sentences_list)==len(tags_list))

len(sentences_list[15]) == len(tags_list[15])


len(sentences_list[50]) == len(tags_list[50])


len(sentences_list[30]) == len(tags_list[30])


len(sentences_list[0]) == len(tags_list[0]) # Collins you are good to build


tokeniser= tf.keras.preprocessing.text.Tokenizer(lower=False,filters='')
tokeniser.fit_on_texts(sentences_list)


print("Vocabulary size of Tokeniser ",len(tokeniser.word_index)+1) # Adding one since 0 is reserved for padding


tokeniser.index_word[15]

encoded_sentence=tokeniser.texts_to_sequences(sentences_list)
#print("First Original Sentence: \n", sentences_list[0])
#print("First Encoded Sentence:\n", encoded_sentence[0])
#print("Is Length of Original Sentence Same as Encoded Sentence: ",len(sentences_list[0])==len(encoded_sentence[0]))
#print("Length of First Sentence: ", len(encoded_sentence[0]))

tags=list(set(data["tags"].values))
#print(tags)

num_tags=len(tags)
print("Number of Tags: ",num_tags)

tags_map={tag:i for i,tag in enumerate(tags)}
#print("Tags Map: ",tags_map)

reverse_tag_map={v: k for k, v in tags_map.items()}

encoded_tags=[[tags_map[w] for w in tag] for tag in tags_list]
#print("First Sentence:\n",sentences_list[0])
#print('First Sentence Original Tags:\n',tags_list[0])
#print("First Sentence Encoded Tags:\n ",encoded_tags[0])
#print("Is length of Original Tags and Encoded Tags same: ",len(tags_list[0])==len(encoded_tags[0]))
#print("Length of Tags for First Sentence: ",len(encoded_tags[0]))


#plt.hist([len(sen) for sen in sentences_list], bins= 100)
#plt.show()


max_sentence_length=max([len(s) for s in sentences_list])
#print(max_sentence_length)

tags_map


max_len = 128
padded_encoded_sentences = pad_sequences(maxlen=max_len,sequences=encoded_sentence,padding="post",value=0)

padded_encoded_tags=pad_sequences(maxlen=max_len,sequences=encoded_tags,padding="post",value=0)

#print("Shape of Encoded Sentence: ",padded_encoded_sentences.shape)
#print("Shape of Encoded Labels: ",padded_encoded_tags.shape)
#print("First Encoded Sentence Without Padding:\n",encoded_sentence[0])
#print("First Encoded Sentence with padding:\n",padded_encoded_sentences[0])
#print("First Sentence Encoded Label without Padding:\n",encoded_tags[0])
#print("First Sentence Encoded Label with Padding:\n",padded_encoded_tags[0])

target= [to_categorical(i,num_classes = num_tags) for i in  padded_encoded_tags]
#print("Shape of Labels  after converting to Categorical for first sentence: ",target[0].shape)
# ### Splitting The Data

X_train,X_val_test,y_train,y_val_test = train_test_split(padded_encoded_sentences,target,test_size = 0.2,random_state=False)
X_val,X_test,y_val,y_test = train_test_split(X_val_test,y_val_test,test_size = 0.2,random_state=False)
print("Input Train Data Shape: ",X_train.shape)
print("Train Labels Length: ",len(y_train))
print("Input Test Data Shape: ",X_test.shape)
print("Test Labels Length: ",len(y_test))

print("Input Validation Data Shape: ",X_val.shape)
print("Validation Labels Length: ",len(y_val))
# In[41]:
#print("First Sentence in Training Data: ",X_train[0])
#print("First sentence Label: ",y_train[0])
#print("Shape of First Sentence -Train: ",X_train[0].shape)
#print("Shape of First Sentence Label  -Train: ",y_train[0].shape)
# ### Training The Model
# In[42]:
embedding_dim=128
vocab_size=len(tokeniser.word_index)+1
lstm_units=128
max_len=128

input_word = Input(shape = (max_len,))
model = Embedding(input_dim = vocab_size+1,output_dim = embedding_dim,input_length = max_len)(input_word)

model = Bidirectional(LSTM(units=embedding_dim,return_sequences=True))(model)
out = TimeDistributed(Dense(num_tags,activation = 'softmax'))(model)
model = Model(input_word,out)
model.summary()
# In[43]:
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
# In[44]:
get_ipython().run_cell_magic('time', '', "early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=2, mode='min', baseline=None, restore_best_weights=False)\n\ncallbacks = [PlotLossesCallback(), early_stopping]\nhistory = model.fit(\n    X_train,\n    np.array(y_train),\n    validation_data=(X_val,np.array(y_val)),\n    batch_size = 32,\n    epochs = 500,\n    callbacks=callbacks,\n    verbose=1)")
# ### Evaluating The Model
# In[45]:
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
    
    
    return pred_data #,accuracy


# In[47]:

#,accuracy
pred_data = collins_Predictionevaluation(X_test,pred) #,y_test)

# In[48]:
y_pred=pred_data['pred_target_tag'].tolist()
#y_actual=pred_data['actual_target_tag'].tolist()
print("y_pred")

# In[49]:
#print(classification_report(y_actual,y_pred))
# In[50]:
#pred_data.tail(10)


# In[53]:


#print("Accuracy for Collins_MedicalNER Model Test Sample is: ", accuracy)

