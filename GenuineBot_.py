import torchvision
from multiprocessing import Process
import torch as torch
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db as database
import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import os
import logging
import numpy as np
import random as python_random
import tensorflow as tf
import torch
import torchvision
from multiprocessing import Process
from torch import nn
import os
import numpy as np
from decimal import Decimal
from math import ceil
import matplotlib.pyplot as plt
from datasets import load_dataset
import functools
import operator
import os
from tensorflow.keras import *
import keras.backend as K
import string
import copy
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re
import csv
from keras.models import model_from_json
from keras.models import load_model
import pandas as pd
import collections
from nltk.tokenize.treebank import TreebankWordDetokenizer
import json
from firebase_admin import ml

####Main utilities

def getMatch(input1,N):
        p = regex.compile("(\\w+)")
        matches = regex.findall(p,input1)
        if len(matches) == 0:
            return ""
        return matches[N-1]
    
def listener(event,dataset):
    nodes = event.data
    if nodes != None:
        nodes = nodes.get().val()
    #if nodes are empty, exit
    if nodes == None:
        return None
    for child in nodes:
        #Match messages
        if getMatch(child.key(),1) == "Message":
            #Add it's content into the text_input
            if getMatch(child.getkey(),2) != "":
                #Get the messages
                text_input = getMatch(child.key(),2)
                dataset.append(text_input)

def get_dataset(dataset):
    path = "C:\\Users\\Administrator\\Downloads\\HeyBoss.json"
    databaseURL = "https://genuine-a483a-default-rtdb.firebaseio.com/"
    cred_obj = firebase_admin.credentials.Certificate(path)
    default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':databaseURL})
    path_to_dataset = "/Dataset/"
    nodes = None
    nodes = database.reference(path_to_dataset).listen(listener(dataset))
    return dataset
              
def calc_distance(message_pred,message):
    word_right = 0
    for i in range(len(message)):
        if message_pred[i] == message[i]:
            word_right += 1
    #Calculate the percentage
    return word_right/len(message)

def transfer_learning(model):
    #Remove the last item
    model.layers.remove(len(model.layers)-1)
    return model

def underfits(average_error,error_thresold):

    if average_error <= error_thresold:
        return True
    else:
        return False


#Get new knowledge
def sentence_list(textdata):
    
    textdata1 = [textdata.strip() for sentence in re.split(r'(?>=[.?]\s+)',textdata) if sentence.strip()]

    return textdata1

##Get the current knowledge 
def current_list():
    
    file_path = "C:\\Users\\Administrator\\Downloads\\archive\\articles.csv"
    data = pd.read_csv(file_path)
    textdata = "".join("".join(line) for line in data['text'])
    textdata1 = [textdata.strip() for sentence in re.split(r'(?>=[.?]\s+)',textdata) if sentence.strip()]

    return textdata1

def evaluate(model,input1,training_set,output_length):
    next_words = output_length
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_set)
    total_words = len(tokenizer.word_index) + 1
    next_sentence = ""
    max_sequence_len = 10
    puncts = string.punctuation
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([input1])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,padding='pre')
        predicted_probs = model(token_list)
        predicted_probs = np.array(predicted_probs)
        predicted_word = puncts[np.argmax(predicted_probs)]
        input1 += " " + predicted_word
        next_sentence += predicted_word + " "
    return next_sentence

#A training algorithm adapted to decrease overfitting and underfitting during production stage
def train(model,training_set,vocab,pred_thresold,error_thresold,training_set2):
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocab)
    total_words = len(tokenizer.word_index) + 1
    PATH = "C:\\Users\\Administrator\\Downloads\\Intermediate_Model_.keras"
    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=1)
    # Create input sequences
    input_sequences = []
    max_length = 1000
    for line in training_set:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            if i < max_length:
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
            elif i == max_length:
                break
    print("Done!")
    # Pad sequences and split into predictors and label
    max_sequence_len = 10
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    for i in range(int(len(X)/pred_thresold)):
        history = model.fit(X[:pred_thresold*(i+1)],y[:pred_thresold*(i+1)],epochs=500,callbacks=[cp_callback])
        accuracy = history.history['accuracy']
        if underfits(accuracy,error_thresold):
            #Then, remove a hidden layer, if possible
            #First, check if this is possible
            if len(model.layers) >= 2:
                #Remove a hidden layer, starting with the most deep layer
                model.layers.remove(len(model.layers)-2)
    #In addition, it's important to re-train the model on its dataset
    #I imagine I won't need to detect signs of overfitting for this stage...
    for line in training_set2:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(token_list)):
            if i < max_length:
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
            elif i == max_length:
                break
    max_sequence_len = 10
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    model.compile(X,y,epochs=500,callbacks=[cp_callback])
    model.save(PATH)
    return tokenizer

#Data represents the examples that have been inferered, which will be used for training
def overfits(inference_count,error_thresold,inference_thresold,cur_error,model,pred,target,data,total,training_set2,database,path,path2,path3):
    history = model.eval(pred,target)
    #Calculate current error
    cur_error += history.history['accuracy']
    error = float(cur_error/inference_count)
    total_outputs = 0
    total_messages = ""
    total = total
    if inference_count >= inference_thresold:
        if error <= error_thresold:
            for message in data:
                total_messages += message + '.' 
            tokenizer2 = Tokenizer()
            total = total + total_messages
            new_data = sentence_list(total_message)
            tokenizer2.fit_on_texts(new_data)
            prev_total = total_outputs
            for word in tokenizer2.word_index:
                for word2 in tokenizer.word_index:
                    if word != word2:
                        total_outputs += 1
            total_outputs += len(tokenizer.word_index) + 1
            if total_outputs > prev_total:
                model = transfer_learning(model)
                model.add(Dense(total_outputs, activation='softmax'))
                pred_thresold = 100
                tokenizer2 = train(model,total_messages,total,pred_thresold,error_thresold,training_set2)
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                converter._experimental_lower_tensor_list_ops = False
                tflite_model = converter.convert()
                savepoint = "Saved_Model.tflite"
                with open(savepoint, 'wb') as f:
                     f.write(tflite_model)
                source = ml.TFLiteGCSModelSource.from_tflite_model_file(savepoint)
                # Create the model object
                tflite_format = ml.TFLiteFormat(model_source=source)  
                model = ml.Model(
                display_name="GenuineTrustPredictor",
                model_format=tflite_format)
                # Add the model to your Firebase project and publish it
                new_model = ml.update_model(model)
                ml.publish_model(new_model.model_id)
                database.reference(path2).set(float(cur_error*100.0))
                database.reference(path3).set(tokenizer2.word_index)
            else:
                #Otherwise, just learn more
                pred_thresold = 100
                train(model,total_messages,total,pred_thresold,error_thresold,training_set2)
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                converter._experimental_lower_tensor_list_ops = False
                tflite_model = converter.convert()
                savepoint = "Saved_Model.tflite"
                with open(savepoint, 'wb') as f:
                     f.write(tflite_model)
                source = ml.TFLiteGCSModelSource.from_tflite_model_file(savepoint)
                # Create the model object
                tflite_format = ml.TFLiteFormat(model_source=source)  
                model = ml.Model(
                display_name="GenuineTrustPredictor",
                model_format=tflite_format)
                # Add the model to your Firebase project and publish it
                new_model = ml.update_model(model)
                ml.publish_model(new_model.model_id)
                database.reference(path2).set(float(cur_error*100.0))
                data.clear()
                inference_count = 0
            return inference_count+1,cur_error,model,tokenizer2,total          
    elif len(data) > inference_thresold:
        inference_count = 0
        #Clear the data
        data.clear()
        return inference_count+1,cur_error,model,tokenizer,total
    else:
        return inference_count+1,cur_error,model,tokenizer,total

#Use my laptop as a Cloud Device
def run_model():

    #Listens to the database in real-time...

    dataset = []
    #Pass in dataset as input...
    if __name__ == "__main__":
        p1 = Process(target=get_dataset,args=(dataset,))
        p1.start()
    average_accuracy = 0
    total_accuracy = 0
    messages_predicted = 0
    thresold = 99.9
    current_data = current_list()
    total = current_data
    max_sequence_len = 10
    tokenizer = Tokenizer()
    everything = current_list()
    tokenizer.fit_on_texts(current_data)
    total_words = len(tokenizer.word_index) + 1
    #######################################Load up the model
    model = Sequential()
    total_puncts = len(string.punctuation)
    #Genuine Bot
    model.add(Embedding(total_words,100,
                    input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(Bidirectional(LSTM(128,return_sequences=False)))
    #Predict the correct type of punctuation
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    checkpoint_path = "training_2/cp.ckpt"
    model.load_weights(checkpoint_path)
    ########################################################
    #Upload the best working model on the firebase real-time database
    path = "C:\\Users\\Administrator\\Downloads\\HeyBoss.json"
    databaseURL = "https://genuine-a483a-default-rtdb.firebaseio.com/"
    cred_obj = firebase_admin.credentials.Certificate(path)
    default_app = firebase_admin.initialize_app(cred_obj,{'databaseURL':'https://genuine-a483a-default-rtdb.firebaseio.com/',
                                                'storageBucket': "genuine-a483a.appspot.com" })
    path2 = "/GenuineTrustAccuracy"
    path3 = "/WordKnowledgeBase"
    #If the error is below or equal 75%...
    error_thresold = 0.75
    cur_error = 0
    inference_count = 1
    inference_thresold = 1000
    pred_thresold = 100
    accumulated_data = []
    cloud_computing = True
    #Send a dictionary
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    savepoint = "Saved_Model.tflite"
    with open(savepoint, 'wb') as f:
        f.write(tflite_model)
    source = ml.TFLiteGCSModelSource.from_tflite_model_file(savepoint)
    # Create the model object
    print("Here!")
    database.reference(path3).set(tokenizer.word_index)
    database.reference(path2).set(0.0)
    while cloud_computing:
        for message in dataset:
            #current data is allowed to change, fully dynamic
            #Strip any punctuation from the message
            message_pred = evaluate(model,message,current_data,len(message.strip().split()))
            accumulated_data.append(message)
            #Monitor for overfitting, underfitting and find solution
            inference_count,cur_error,model,tokenizer,total = overfits(inference_count,error_thresold,inference_thresold,
                                                                       cur_error,message_pred,message,accumulated_data,
                                                                       tokenizer,total,current_data,database,path,path2,path3)
        dataset.clear()
run_model()

import functions_framework

# CloudEvent function to be triggered by an Eventarc Cloud Audit Logging trigger
# Note: this is NOT designed for second-party (Cloud Audit Logs -> Pub/Sub) triggers!
@functions_framework.cloud_event
def hello_auditlog(cloudevent=run_model):
    # Print out the CloudEvent's (required) `type` property
    # See https://github.com/cloudevents/spec/blob/v1.0.1/spec.md#type
    print(f"Event type: {cloudevent['type']}")

    # Print out the CloudEvent's (optional) `subject` property
    # See https://github.com/cloudevents/spec/blob/v1.0.1/spec.md#subject
    if 'subject' in cloudevent:
        # CloudEvent objects don't support `get` operations.
        # Use the `in` operator to verify `subject` is present.
        print(f"Subject: {cloudevent['subject']}")

    # Print out details from the `protoPayload`
    # This field encapsulates a Cloud Audit Logging entry
    # See https://cloud.google.com/logging/docs/audit#audit_log_entry_structure

    payload = cloudevent.data.get("protoPayload")
    if payload:
        print(f"API method: {payload.get('methodName')}")
        print(f"Resource name: {payload.get('resourceName')}")
        print(f"Principal: {payload.get('authenticationInfo', dict()).get('principalEmail')}")
