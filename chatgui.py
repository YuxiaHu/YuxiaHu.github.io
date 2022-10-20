# -*- coding: utf-8 -*-
# @Time     : Nov.2020
# @Email    : yuxhu@uni-osnabrueck.de
# @FileName : chatgui.py
# Part of the content is based on https://data-flair.training/blogs/python-chatbot-project/

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

import json
import pickle
import time
import numpy as np

from sklearn import preprocessing, metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from model_textcnn_lstm_FP import load_file, preprocess, train_model
from model_textcnn_lstm_FP import ignore_stopwords_and_lemmatize


# ------------- tokenizer and encoder for s, f and p -----------
s_data = load_file('intents_S.json')
s_sentences, s_labels = s_data[0], s_data[1]
s_preproce = preprocess(s_sentences, s_labels, 88)
s_tokenizer, s_encoder = s_preproce[4], s_preproce[6]

f_data = load_file('intents_F.json')
f_sentences, f_labels = f_data[0], f_data[1]
f_preproce = preprocess(f_sentences, f_labels, 2)
f_tokenizer, f_encoder = f_preproce[4], f_preproce[6]

p_data = load_file('intents_P.json')
p_sentences, p_labels = p_data[0], p_data[1]
p_preproce = preprocess(p_sentences, p_labels, 2)
p_tokenizer, p_encoder = p_preproce[4], p_preproce[6]

# ---------------------------------------------------------------

# global variables that determine whether to 
# update the training data in the json file
to_update_json_s = False
to_update_json_f = False
to_update_json_p = False

# global variables that determine 
# whether to update the model
to_update_model_s = False
to_update_model_f = False
to_update_model_p = False

# load the models textcnn，lstm and textcnn + lstm
# s_textcnn_model = load_model('model_textcnn_S.h5')
# f_textcnn_model = load_model('model_textcnn_F.h5')
# p_textcnn_model = load_model('model_textcnn_P.h5')

# s_lstm_model = load_model('model_lstm_S.h5')
# f_lstm_model = load_model('model_lstm_F.h5')
# p_lstm_model = load_model('model_lstm_P.h5')

s_textcnn_lstm_model = load_model('model_textcnn_lstm_S.h5')
f_textcnn_lstm_model = load_model('model_textcnn_lstm_F.h5')
p_textcnn_lstm_model = load_model('model_textcnn_lstm_P.h5')

# load the list that contains all the origin labels
hazard_origins = pickle.load(open('hazard_origin_labels.pkl','rb'))
# load the s json file
intents_S = json.loads(open('intents_S.json').read())


# # load the models
# count_model_S = load_model('count_model_S.h5')
# tfidf_model_S = load_model('tfidf_model_S.h5')
# count_model_F = load_model('count_model_F.h5')
# count_model_P = load_model('count_model_P.h5')

# intents_S = json.loads(open('intents_S.json').read())
# intents_F = json.loads(open('intents_F.json').read())
# intents_P = json.loads(open('intents_P.json').read())

# training_S = TrainModel('intents_S.json')
# training_S.data_vectorize()

# encoder_s = training_S.encoder
# count_vect_s = training_S.count_vect
# tfidf_vect = training_S.tfidf_vect

# training_F = TrainModel('intents_F.json')
# training_F.data_vectorize()

# encoder_f = training_F.encoder
# count_vect_f = training_F.count_vect

# training_P = TrainModel('intents_P.json')
# training_P.data_vectorize()

# encoder_p = training_P.encoder
# count_vect_p = training_P.count_vect

# The number of words with the highest frequency extracted 
# when comparing the similarity of two sentences
NUMBER_OF_KEYWORDS = 7


# def load_classifier(file_name):
#     with open(file_name, 'rb') as f:
#         classifier = pickle.load(f)
#         return classifier

# # load the machine learning classifier
# naive_bayes_count_classifier = load_classifier('naive_bayes_count_classifier.pkl')
# naive_bayes_tfidf_classifier = load_classifier('naive_bayes_tfidf_classifier.pkl')
# decision_tree_count_classifier = load_classifier('decision_tree_count_classifier.pkl')
# decision_tree_tfidf_classifier = load_classifier('decision_tree_tfidf_classifier.pkl')
# random_forest_count_classifier = load_classifier('random_forest_count_classifier.pkl')
# random_forest_tfidf_classifier = load_classifier('random_forest_tfidf_classifier.pkl')


def ignore_stopwords(sentence):
    stopwords_list = stopwords.words('english')
    for w in ['!',',','.','?','+','-']:
        stopwords_list.append(w)
    # remove all stop words in the sentence
    filtered_words = [word for word in nltk.word_tokenize(sentence) if word not in stopwords_list]
    return filtered_words


def stemming_sentence(sentence):
    filtered_words = ignore_stopwords(sentence)
    # stemming
    porter_words = [porter.stem(w.lower()) for w in filtered_words]
    # words list -> sentence
    filtered_sentence = ' '.join(porter_words)
    print ("\nThe original sentence is: " + sentence)
    print ("The sentence after stemming is: " + filtered_sentence + '\n')
    return filtered_sentence


def lemmatize_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    # stemming
    lemmatize_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # words list -> sentence
    filtered_sentence = ' '.join(lemmatize_words)
    # print ("\nThe original sentence is: " + sentence)
    # print ("The sentence after stemming is: " + filtered_sentence + '\n')
    return filtered_sentence


# predict which tag the sentence belongs to with deep learning
def dl_predict_class_s(sentence, model, tokenizer, encoder):

    filtered_sentence = ignore_stopwords_and_lemmatize(sentence)
    # convert words to a list of numbers
    x_test_word_ids = tokenizer.texts_to_sequences([filtered_sentence])
    # set the length of each sample to a fixed value
    # cut off the part that exceeds the fixed value and fill in the part with 0 at the top
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    res = model.predict(x_test_padded_seqs)[0]
    res_label_index = np.argmax(res, axis=-1)

    # filter out predictions below a threshold
    ERROR_THRESHOLD_1 = 0.5
    ERROR_THRESHOLD_2 = 0.01

    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD_2]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    # print ("----------------------- predict! -----------------------")
    # print (results)
    # print (np.array(results).shape)
    # print ("--------------------------------------------------------")

    return_list = []
    # return one label if the probability is greater than 0.5
    if res[res_label_index] > ERROR_THRESHOLD_1:
        tag = encoder.inverse_transform([res_label_index])
        return_list.append({"label": tag[0], "probability": res[res_label_index]})
    # otherwise add multiple labels above threshold 2
    else:
        for r in results:
            tag = encoder.inverse_transform([r[0]])
            return_list.append({"label": tag[0], "probability": r[1]})

    return return_list


def dl_predict_class_fp(sentence, model, tokenizer, encoder):

    filtered_sentence = ignore_stopwords_and_lemmatize(sentence)
    # convert words to a list of numbers
    x_test_word_ids = tokenizer.texts_to_sequences([filtered_sentence])
    # set the length of each sample to a fixed value
    # cut off the part that exceeds the fixed value and fill in the part with 0 at the top
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)
    res = model.predict(x_test_padded_seqs)[0]
    res_label_index = np.argmax(res, axis=-1)

    # difference between two label probabilities
    DIFFERENCE_THRESHOLD = 0.2

    # list the result index and probabilitiy
    results = [[i,r] for i,r in enumerate(res)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    # print ("----------------------- predict! -----------------------")
    # print (results)
    # print ("--------------------------------------------------------")

    return_list = []
    # choose the one with the highest probability 
    # if the difference is greater than 0.2
    if abs(res[0] - res[1]) > DIFFERENCE_THRESHOLD:
        tag = encoder.inverse_transform([res_label_index])
        return_list.append({"label": tag[0], "probability": res[res_label_index]})
    # otherwise add both
    else:
        for r in results:
            tag = encoder.inverse_transform([r[0]])
            return_list.append({"label": tag[0], "probability": r[1]})

    return return_list



# predict which tag the sentence belongs to with machine learning
def ml_predict_class(sentence, classifier, vectorizer, encoder):

    filtered_sentence = lemmatize_sentence(sentence)
    p = vectorizer.transform([filtered_sentence]).toarray()
    res = classifier.predict_proba(p)[0]
    index = np.argmax(res, axis=-1)

    # filter out predictions below a threshold
    ERROR_THRESHOLD = 0.2
    if res[index] > ERROR_THRESHOLD:
        tag = encoder.inverse_transform([index])
    else:
        tag = [0]
    return tag[0]



# get the words with the highest frequency
def get_freq_dist(text1, text2):
    all_text = text1 + " " + text2
    words = nltk.word_tokenize(all_text)
    freq_dist = FreqDist(words)
    most_common_words = freq_dist.most_common(NUMBER_OF_KEYWORDS)
    return most_common_words

# mark the position of each keyword with a dictionary
def find_position(common_words):  
    result = {}
    pos = 0
    for word in common_words:
        result[word[0]] = pos
        pos += 1
    return result

# convert text into word frequency vector
def text_to_vector(words, pos_dict):
    freq_vec = [0] * NUMBER_OF_KEYWORDS
    for word in words:
        if word in list(pos_dict.keys()):
            freq_vec[pos_dict[word]] += 1
    return freq_vec   


# calculate the cosine similarity, close to 1 -> the similarity is high
def cosine_similarity(x,y):
    # ignore "divide by zero" or "divide by NaN"
    np.seterr(divide='ignore', invalid='ignore')
    num = np.dot(x, y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


# according to the sentence to determine which pattern it belongs to
# def getPattern(sentence, ints):
#     sentence_filtered = stemming_sentence(sentence)
#     tag = ints[0]['intent']

#     cosine_similar_max = 0
#     pattern_max = 0

#     for intent in intents_S['intents']:
#         if intent['tag'] == tag:
#             for pattern in intent['patterns']:
                
#                 pattern_filtered = stemming_sentence(pattern)
#                 # find the frequently used words
#                 most_common_words = get_freq_dist(pattern_filtered, sentence_filtered)
#                 # record the position of frequently used words
#                 pos_dict = find_position(most_common_words)
#                 print ("The common words between pattern and user input: ")
#                 print (str(most_common_words))

#                 # convert pattern and sentence into word frequency vectors
#                 pattern_vec = text_to_vector(nltk.word_tokenize(pattern_filtered), pos_dict)
#                 print ("The word frequency vectors of the pattern: " + str(pattern_vec))
#                 sentence_vec = text_to_vector(nltk.word_tokenize(sentence_filtered), pos_dict)
#                 print ("The word frequency vectors of the user input: " + str(sentence_vec))
#                 cosine_similar = cosine_similarity(np.array(pattern_vec), np.array(sentence_vec))
#                 print ("The cosine similarity between the two is: " + str(cosine_similar) + '\n\n')

#                 # get the maximum cosine similarity and the corresponding pattern
#                 if cosine_similar > cosine_similar_max:
#                     cosine_similar_max = cosine_similar
#                     pattern_max = pattern

#     return(pattern_max)


# def getConsequences(sentence, ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     # list to save all potential consequences and severity levels
#     consequences = []

#     for i in list_of_intents:
#         if(i['tag']== tag):
#             pattern = getPattern(sentence, ints)
#             # if the user input has no cosine similarity with all sources in this tag
#             # most likely it is because the predicted class is wrong, that is, 
#             # the incorrect tag is obtained.
#             if pattern == 0:
#                 print ("The predicted class based on user input is " + tag)
#                 print ("No Result")
#                 print ("-------------------------------------------------------------------------------")
#                 return 0
#             print ("The predicted class based on user input is " + tag)
#             print ("The pattern we get is: " + pattern)

#             list_of_responses = i['responses']
#             print ("The following potential consequences and severity levels were found: ")
#             for array in list_of_responses:
#                 for origin in array[2:len(array)]: # origins start from the third position
#                     if origin == pattern:
#                         consequences.append((array[0], array[1]))
#                         print (array[0], array[1])
#             break
#     print ("-------------------------------------------------------------------------------")
#     return consequences


def getPLr(s, f, p):
    if s == 'S1':
        if f == 'F1':
            if p == 'P1':
                plr = '(PLr-a)'
            else:
                plr = '(PLr-b)'
        else:
            if p == 'P1':
                plr = '(PLr-b)'
            else:
                plr = '(PLr-c)'
    else:
        if f == 'F1':
            if p == 'P1':
                plr = '(PLr-c)'
            else:
                plr = '(PLr-d)'
        else:
            if p == 'P1':
                plr = '(PLr-d)'
            else:
                plr = '(PLr-e)'
    return plr



def getConsequences(tags, intents_json):
    list_of_intents = intents_json['intents']
    # list to save all potential consequences and severity levels
    consequences = []

    # no tag found above the threshold
    if not tags:
        return 0

    for tag in tags:
        for intent in list_of_intents:
            if intent['tag'] == tag['label']:
                responses = []
                for respo in intent['responses']:
                    responses.append(respo)
                consequences.append({"origin": intent['tag'], "s_level": responses})
    return consequences



def chatbot_response(msg, option_select, conseq_select, possible_origin_select, origin_select, flevel_select, plevel_select):

    # predict class with textcnn model
    # s_tag_textcnn = dl_predict_class_s(msg, s_textcnn_model, s_tokenizer, s_encoder)
    # f_tag_textcnn = dl_predict_class_fp(msg, f_textcnn_model, f_tokenizer, f_encoder)
    # p_tag_textcnn = dl_predict_class_fp(msg, p_textcnn_model, p_tokenizer, p_encoder)

    # # predict class with lstm model
    # s_tag_lstm = dl_predict_class_s(msg, s_lstm_model, s_tokenizer, s_encoder)
    # f_tag_lstm = dl_predict_class_fp(msg, f_lstm_model, f_tokenizer, f_encoder)
    # p_tag_lstm = dl_predict_class_fp(msg, p_lstm_model, p_tokenizer, p_encoder)

    # predict class with textcnn + lstm model
    s_tag_textcnn_lstm = dl_predict_class_s(msg, s_textcnn_lstm_model, s_tokenizer, s_encoder)
    f_tag_textcnn_lstm = dl_predict_class_fp(msg, f_textcnn_lstm_model, f_tokenizer, f_encoder)
    p_tag_textcnn_lstm = dl_predict_class_fp(msg, p_textcnn_lstm_model, p_tokenizer, p_encoder)


    # # predict class with different models and vectorizers
    # dl_count_tag_S = dl_predict_class(msg, count_model_S, count_vect_s, encoder_s)
    # dl_count_tag_F = dl_predict_class(msg, count_model_F, count_vect_f, encoder_f)
    # dl_count_tag_P = dl_predict_class(msg, count_model_P, count_vect_p, encoder_p)
    # dl_tfidf_tag_S = dl_predict_class(msg, tfidf_model_S, tfidf_vect, encoder_s)

    # # naive bayes classifier and count vectorizer
    # nb_count_tag_S = ml_predict_class(msg, naive_bayes_count_classifier, count_vect_s, encoder_s)

    # # naive bayes classifier and tfidf vectorizer
    # nb_tfidf_tag_S = ml_predict_class(msg, naive_bayes_tfidf_classifier, tfidf_vect, encoder_s)

    # # decision tree classifier and count vectorizer
    # dt_count_tag_S = ml_predict_class(msg, decision_tree_count_classifier, count_vect_s, encoder_s)

    # # decision tree classifier and tfidf vectorizer
    # dt_tfidf_tag_S = ml_predict_class(msg, decision_tree_tfidf_classifier, tfidf_vect, encoder_s)

    # # random forest classifier and count vectorizer
    # rf_count_tag_S = ml_predict_class(msg, random_forest_count_classifier, count_vect_s, encoder_s)

    # # random forest classifier and tfidf vectorizer
    # rf_tfidf_tag_S = ml_predict_class(msg, random_forest_tfidf_classifier, tfidf_vect, encoder_s)


    # s_tag_list = s_tag_textcnn
    # f_level_list = f_tag_textcnn
    # p_level_list = p_tag_textcnn

    # s_tag_list = s_tag_lstm
    # f_level_list = f_tag_lstm
    # p_level_list = p_tag_lstm

    s_tag_list = s_tag_textcnn_lstm
    f_level_list = f_tag_textcnn_lstm
    p_level_list = p_tag_textcnn_lstm

    global to_update_json_s
    global to_update_json_f
    global to_update_json_p

    global to_update_model_s
    global to_update_model_f
    global to_update_model_p
    
    # ignore the predicted results if the user gives the choices
    if origin_select:
        s_tag_list = origin_select
    if flevel_select:
        f_level_list = flevel_select
    if plevel_select:
        p_level_list = plevel_select
    
    consequences_list = getConsequences(tags=s_tag_list , intents_json=intents_S)

    # provide the user with a list of all hazards to choose from
    # if no prediction result is above the threshold
    if consequences_list == 0:
        result = "I have not found a suitable hazard origin. So I provide the following hazard origin list for you.\n\n"
        for i, origin in enumerate(hazard_origins):
            result += str(i+1) + ". " + origin + "\n"
        result += "\nYou can enter the number before the hazard origin to select."
        step_info = 'origin_list_select'
        return_result = {'result': result, 'step': step_info}


    # if there is a prediction result
    elif len(consequences_list) == 1:
        origin = consequences_list[0]['origin']
        responses = consequences_list[0]['s_level']

        # if the hazard origin contains only one potential consequence
        if len(responses) == 1:
            if option_select == 'a':
                # if the f and p level predictions are ambiguous 
                # need to be selected by the user and to update the file
                if not len(f_level_list) == 1:
                    result = user_flevel_select()
                    step_info = 'f_level_select'
                    return_result = {'result': result, 'step': step_info}
                    to_update_json_f = True
                elif not len(p_level_list) == 1:
                    result = user_plevel_select()
                    step_info = 'p_level_select'
                    return_result = {'result': result, 'step': step_info}
                    to_update_json_p = True
                else:
                    print ('---------- responses length is 1! ----------')
                    f_level = f_level_list[0]['label']
                    p_level = p_level_list[0]['label']
                    result1 = "1. The potential consequence for this hazard scenario is: " + responses[0][0]
                    result2 = "\n2. The required performance level of the safety function is: "
                    s_level = responses[0][1]
                    plr = getPLr(s_level, f_level, p_level)
                    result = result1 + result2 + s_level + " -> " + f_level + " -> " + p_level + " -> " + plr
                    step_info = 'new_hazard_input'
                    return_result = {'result': result, 'step': step_info}
            elif option_select == 'b':
                result = "The potential consequence for this hazard origin is: " + responses[0][0]
                step_info = 'new_hazard_input'
                return_result = {'result': result, 'step': step_info}
        
        # if the origin contains multiple potential consequences
        else:
            if option_select == 'b':
                result = "The potential consequences for this hazard origin are: "
                for r in responses[0:len(responses)-1]:
                    result += r[0] + ', '
                last = responses.pop()
                result += last[0]
                step_info = 'new_hazard_input'
                return_result = {'result': result, 'step': step_info}

            elif option_select == 'a':
                # the user has not chosen which consequence to keep
                # list all the consequences
                if conseq_select == 'null':
                    print ('---------- conseq_select == null! ----------')
                    result = "The hazard origin you input is " + origin + ". For this I have found the following potential consequences. Which one of these is most likely? Or do you want to keep all of them? \n\n"
                    for i, r in enumerate(responses):
                        result += str(i+1) + ". " + r[0] + " (" + r[1] + ")\n"
                    result += str((len(responses) + 1)) + ". keep all"
                    step_info = 'conseq_select'
                    return_result = {'result': result, 'step': step_info}

                else:
                    print ('---------- conseq_select != null! ----------')
                    if not conseq_msg_true(responses, conseq_select):
                        result = "Please input a correct number (1 - " + str(len(responses) + 1) + ")."
                        step_info = 'conseq_select'
                        return_result = {'result': result, 'step': step_info}
                    if not len(f_level_list) == 1:
                        result = user_flevel_select()
                        step_info = 'f_level_select'
                        return_result = {'result': result, 'step': step_info}
                        to_update_json_f = True
                    elif not len(p_level_list) == 1:
                        result = user_plevel_select()
                        step_info = 'p_level_select'
                        return_result = {'result': result, 'step': step_info}
                        to_update_json_p = True
                    else:
                        f_level = f_level_list[0]['label']
                        p_level = p_level_list[0]['label']

                        if int(conseq_select) > len(responses):
                            print ('---------- conseq_select is keep all ----------')
                            result1 = "1. The potential consequences for this hazard scenario are: "
                            result2 = "\n2. The required performance levels of safety functions are: "
                            for r in responses[0:len(responses)-1]:
                                s_level = r[1]
                                plr = getPLr(s_level, f_level, p_level)
                                result1 = result1 + r[0] + ', '
                                result2 = result2 + s_level + " -> " + f_level + " -> " + p_level + " -> " + plr + ",  "

                            last = responses.pop()
                            s_level = last[1]
                            plr = getPLr(s_level, f_level, p_level)
                            result1 = result1 + last[0]
                            result2 = result2 + s_level + " -> " + f_level + " -> " + p_level + " -> " + plr
                            result = result1 + result2
                        else:
                            print ('---------- conseq_select is not keep all ----------')
                            for i, r in enumerate(responses):
                                if (i+1) == int(conseq_select):
                                    result = "For the potential consequence " + r[0] + " the required performance level of the safety function is: "
                                    s_level = r[1]
                                    plr = getPLr(s_level, f_level, p_level)
                                    result += s_level + " -> " + f_level + " -> " + p_level + " -> " + plr

                        step_info = 'new_hazard_input'
                        return_result = {'result': result, 'step': step_info}

    # if there are multiple prediction results with low probabilities
    else:
        # the user has not selected which origin is desired
        # list all the possible origins
        if possible_origin_select == 'null':
            result = "I have found the following possible hazard origins.\n\n"
            for i, con in enumerate(consequences_list):
                result += str(i+1) + ". " + con['origin'] + "\n"
            result += "\nIs there a hazard origin you want on the list? If yes, please enter the number before the hazard origin to select. If not, enter 0."
            step_info = 'possible_origin_select'
            return_result = {'result': result, 'step': step_info}
            to_update_json_s = True
        else:
            if not possible_origin_msg_true(consequences_list, possible_origin_select):
                result = "Please input a correct number (0 - " + str(len(consequences_list)) + ")."
                step_info = 'possible_origin_select'
                return_result = {'result': result, 'step': step_info}
            
            # if all possible origins are incorrect, give the list for the user to select
            elif possible_origin_select == '0':
                result = "I have the following hazard origin list for you.\n\n"
                for i, origin in enumerate(hazard_origins):
                    result += str(i+1) + ". " + origin + "\n"
                result += "\nYou can enter the number before the hazard origin to select."
                step_info = 'origin_list_select'
                return_result = {'result': result, 'step': step_info}
            # user selected one of them
            else:
                origins = []
                for i, con in enumerate(consequences_list):
                    if (i+1) == int(possible_origin_select):
                        origins.append({"label": con['origin'], "probability": 1})

                        # update the json file
                        if to_update_json_s:
                            write_in_s_file = write_new_msg_in_json_file(msg, origins, 'intents_S.json')
                            if write_in_s_file:
                                to_update_model_s = True
                                to_update_json_s = False
                        return_result = chatbot_response(msg, option_select, conseq_select, 
                                                         possible_origin_select, origins, 
                                                         flevel_select, plevel_select)
       
    return return_result


def user_origin_select(origin_list_select):
    origins = []
    for i, origin in enumerate(hazard_origins):
        if (i+1) == int(origin_list_select):
            origins.append({"label": origin, "probability": 1})
    return origins

def user_flevel_select():
    result = "How often does the hazard occur or how long is the exposure to the hazard?\n\n"
    result += "1. F1 -- seldom to less ofen and/or exposure time is short\n"
    result += "2. F2 -- frequent to continuous and/or exposure time is long\n"
    result += "\nInput 1 or 2 to select"
    return result

def user_plevel_select():
    result = "What is the possibility of avoiding hazard or limiting harm?\n\n"
    result += "1. P1 -- possible under specific conditions\n"
    result += "2. P2 -- scarcely possible\n"
    result += "\nInput 1 or 2 to select"
    return result

# verify the correctness of user input
def origin_list_msg_true(msg):
    try:
        if int(msg) in range(1, 89):
            return True
    except Exception:
        return False

def conseq_msg_true(res_list, msg):
    try:
        if int(msg) in range(1, len(res_list) + 2):
            return True
    except Exception:
        return False

def possible_origin_msg_true(conseq_list, msg):
    try:
        if int(msg) in range(0, len(conseq_list) + 1):
            return True
    except Exception:
        return False

# write the new data marked with star into the json file
def write_new_msg_in_json_file(msg, tag, file_name):
    msg = "* " + msg 
    json_file = json.loads(open(file_name).read())

    i = 0
    try:
        for intent in json_file['intents']:
            if tag[0]['label'] == intent['tag']:
                patterns = intent['patterns']
                patterns.append(msg)
                json_file['intents'][i]['patterns'] = patterns

                with open(file_name, 'w') as write_f:
                    json.dump(json_file, write_f, indent=2, ensure_ascii=False)
                break
            else:
                i = i + 1
    except Exception:
        print ("########### Error writing to file \"" + file_name + "\"! ###########")
        return False
    else:
        print ("########### Success writing to file \"" + file_name + "\"! ###########")
        return True



# --------------------------------- Creating GUI with tkinter ---------------------------------
import tkinter as tk
from tkinter import *
import tkinter.messagebox
import webbrowser

# the control step variables
step_info = ''
option_select = ''
conseq_select = 'null'
possible_origin_select = 'null'
context_info = ''
msg_true = True

origin_select = 0
flevel_select = 0
plevel_select = 0


# determine whether for a specific few origins
# the user also gives the corresponding hazard group
def hazard_origin_in_two_groups(msg):
    msg = lemmatize_sentence(msg)
    if "cavitation phenomenon" in msg or "scraping surface" in msg or "unbalanced rotating part" in msg or "worn part" in msg:
        if "noise" not in msg and "vibration" not in msg:
            return True

def send():
    # reading data of text box starting from first char till end
    msg = EntryBox.get("1.0",'end-1c').strip()
    # delete from position 0 till end
    EntryBox.delete("0.0",END)
    # update the delete
    EntryBox.update()
    # keep the blinking cursor inside text box
    EntryBox.focus()

    global step_info
    global option_select
    global conseq_select
    global possible_origin_select
    global context_info
    global msg_true
    global origin_select
    global flevel_select
    global plevel_select

    global to_update_json_s
    global to_update_json_f
    global to_update_json_p
    global to_update_model_s
    global to_update_model_f
    global to_update_model_p

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You:", "you")
        ChatLog.tag_configure("you", font = ("Bradley Hand", 12), underline=True)
        ChatLog.insert(END, '\n' + msg + '\n\n')

        ChatLog.insert(END, "Bot:", "bot")
        ChatLog.tag_configure("bot", font = ("Bradley Hand", 12), underline=True)

        if msg == 'a' or msg =='(a)':
            ChatLog.insert(END, "\nPlease provide your hazard scenario.\n\n")
            option_select = 'a'
            conseq_select = 'null'
            possible_origin_select = 'null'
            origin_select = 0
            flevel_select = 0
            plevel_select = 0
            step_info == 'new_hazard_input'

        elif msg == 'b' or msg =='(b)':
            ChatLog.insert(END, "\nPlease provide your hazard origin.\n\n")
            option_select = 'b'
            conseq_select = 'null'
            possible_origin_select = 'null'
            origin_select = 0
            flevel_select = 0
            plevel_select = 0
            step_info == 'new_hazard_input'

        elif hazard_origin_in_two_groups(msg):
            ChatLog.insert(END, "\nPlease enter 1 or 2 to select noise-related or vibration-related.\n\n")
            step_info = 'hazard_origin_in_two_groups_select'
            context_info = msg

        else:
            if option_select:
                if step_info == 'origin_list_select':
                    if origin_select == 0:
                        if origin_list_msg_true(msg):
                            origin_select = user_origin_select(msg)
                            res = user_flevel_select()
                        else:
                            res = "Please input a correct number (1 - 88)."
                    elif flevel_select == 0:
                        if msg == '1':
                            flevel_select = [{"label": 'F1', "probability": 1}]
                        elif msg == '2':
                            flevel_select = [{"label": 'F2', "probability": 1}]
                        if flevel_select:
                            res = user_plevel_select()
                        else:
                            res = "Please input a correct number (1 or 2)."
                    elif plevel_select == 0:
                        if msg == '1':
                            plevel_select = [{"label": 'P1', "probability": 1}]
                        elif msg == '2':
                            plevel_select = [{"label": 'P2', "probability": 1}]
                        if plevel_select:
                            res = chatbot_response(context_info, option_select, conseq_select, 
                                                   possible_origin_select, origin_select, 
                                                   flevel_select, plevel_select)

                            # the user input should be written to the file after getting the corresponding origin, f and p values
                            write_in_s_file = write_new_msg_in_json_file(context_info, origin_select, 'intents_S.json')
                            write_in_f_file = write_new_msg_in_json_file(context_info, origin_select, 'intents_F.json')
                            write_in_p_file = write_new_msg_in_json_file(context_info, origin_select, 'intents_P.json')

                            # the model should also be updated at the end if the file is updated
                            if write_in_s_file:
                                to_update_model_s = True
                            if write_in_f_file:
                                to_update_model_f = True
                            if write_in_p_file:
                                to_update_model_p = True

                            step_info = res['step']
                            res = res['result']
                            if step_info == 'new_hazard_input':
                                origin_select = 0
                                flevel_select = 0
                                plevel_select = 0
                            print ("---------- first function! ----------")
                            print ("---------- step_info: " + str(step_info) + " ----------")
                        else:
                            res = "Please input a correct number (1 or 2)."
                    ChatLog.insert(END, '\n' + res + '\n\n')

                else:
                    res = ''
                    if step_info == 'conseq_select':
                        conseq_select = msg
                        msg = context_info
                        print ("conseq_select!")
                    elif step_info == 'possible_origin_select':
                        possible_origin_select = msg
                        msg = context_info
                        print ("possible_origin_select!")
                    elif step_info == 'f_level_select':
                        if msg == '1':
                            flevel_select = [{"label": 'F1', "probability": 1}]
                        elif msg == '2':
                            flevel_select = [{"label": 'F2', "probability": 1}]
                        if not flevel_select:
                            msg_true = False
                            res = "Please input a correct number (1 or 2)."
                        else:
                            if to_update_json_f:
                                write_in_f_file = write_new_msg_in_json_file(context_info, flevel_select, 'intents_F.json')
                                if write_in_f_file:
                                    to_update_model_f = True
                                    to_update_json_f = False
                        msg = context_info
                        print ("flevel_select!")
                    elif step_info == 'p_level_select':
                        if msg == '1':
                            plevel_select = [{"label": 'P1', "probability": 1}]
                        elif msg == '2':
                            plevel_select = [{"label": 'P2', "probability": 1}]
                        if not plevel_select:
                            msg_true = False
                            res = "Please input a correct number (1 or 2)."
                        else:
                            if to_update_json_p:
                                write_in_p_file = write_new_msg_in_json_file(context_info, plevel_select, 'intents_P.json')
                                if write_in_p_file:
                                    to_update_model_p = True
                                    to_update_json_p = False
                        msg = context_info
                        print ("plevel_select!")
                    elif step_info == 'hazard_origin_in_two_groups_select':
                        if msg == '1':
                            msg = context_info + " noise hazards"
                        elif msg == '2':
                            msg = context_info + " vibration hazards"
                        else:
                            msg_true = False
                            res = "Please input a correct number (1 or 2)."
                        print ("hazard_origin_in_two_groups_select!")

                    print ("---------- second function! ----------")
                    print ("step_info: " + str(step_info))
                    print ('msg: ' + msg)
                    print ('option_select: ' + str(option_select))
                    print ('conseq_select: ' + str(conseq_select))
                    print ('possible_origin_select: ' + str(possible_origin_select))
                    print ('origin_select: ' + str(origin_select))
                    print ('flevel_select: ' + str(flevel_select))
                    print ('plevel_select: ' + str(plevel_select))

                    if msg_true:
                        res = chatbot_response(msg, option_select, conseq_select, 
                                               possible_origin_select, origin_select, 
                                               flevel_select, plevel_select)
                        step_info = res['step']
                        if step_info == 'new_hazard_input':
                            conseq_select = 'null'
                            possible_origin_select = 'null'
                            origin_select = 0
                            flevel_select = 0
                            plevel_select = 0
                        context_info = msg
                        print ("---------- step_info: " + str(step_info) + " ----------")
                        ChatLog.insert(END, '\n' + res['result'] + '\n\n')
                    else:
                        msg_true = True
                        ChatLog.insert(END, '\n' + res + '\n\n')
            else:
                ChatLog.insert(END, "\nPlease type the option first." + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def link_show_hand_cursor(event):  
    ChatLog.config(cursor='arrow')  
  
def link_show_xterm_cursor(event):  
    ChatLog.config(cursor='xterm')  
  
def link_click(event): 
    URL = "https://nobelcert.com/DataFiles/FreeUpload/BS%20EN%20ISO%2012100-2010.pdf"
    webbrowser.open(URL)

def wrap(event):
    EntryBox.insert(END, '')

def closeWindow():
    # confirm the message box before closing the window
    ans = tk.messagebox.askyesno(title="Quit", message="Do you want to quit?")
    if ans:
        win.destroy()
    else:
        return
 
# create window
win = tk.Tk()
win.title("Machinery Safety Chatbot")

screenwidth = win.winfo_screenwidth()
screenheight = win.winfo_screenheight()
winwidth = 650
winheight = 560

# calculate the coordinate in the middle of the screen
x = (screenwidth - winwidth) / 2
y = (screenheight - winheight) / 2
win.geometry("%dx%d+%d+%d" %(winwidth, winheight ,x, y))
win.resizable(width=FALSE, height=FALSE)

# Create chat interface
ChatLog = tk.Text(win, bd=0, bg="white", fg="black", font=("Verdana", 10), wrap=WORD)
ChatLog.insert(END, "\nWelcome to this chatbot. The chatbot will assist you with the following according to ISO 12100:2010 (a free version click")
ChatLog.insert(END, ' here', 'link')
ChatLog.tag_configure('link', foreground='DodgerBlue', underline=True)
ChatLog.insert(END, ") and ISO 13849-1: 2015 \n\n")
ChatLog.insert(END, "Option (a): To determine safety function of a hazard scenario/situation. \n")
ChatLog.insert(END, "Option (b): To determine the potential consequences of a hazard. \n")
ChatLog.insert(END, "You can type \"a\" or \"b\" to select. \n\n")
ChatLog.insert(END, "Usage: send - 'Enter',  wrap - 'Shift + Enter',  quit - 'Esc' \n\n")
ChatLog.insert(END, "--------------------------------------------------------------------------------------------------------------------------\n\n")

# mouse pointing  
ChatLog.tag_bind('link', '<Enter>', link_show_hand_cursor)  
# mouse away
ChatLog.tag_bind('link', '<Leave>', link_show_xterm_cursor)  
# mouse click  
ChatLog.tag_bind('link', '<Button-1>', link_click)  
# disable changes on ChatLog
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = tk.Scrollbar(win, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
# SendButton = tk.Button(win, font=("Arial",10), text="Send", width="6", height="3",
#                        bd=0, bg="#C0C0C0", activebackground="#DCDCDC",fg='black', 
#                        command= send)

# Create the box to enter message
EntryBox = tk.Text(win, bd=0, bg="white", fg="black", font=("Arial, 10"), wrap=WORD)
# keyboard Enter -> trigger event send
EntryBox.bind('<Return>', (lambda event: send()))
# keyboard Shift + Enter -> trigger event wrap
EntryBox.bind('<Shift-Return>', wrap)

# close window by clicking -> trigger event closeWindow
win.protocol("WM_DELETE_WINDOW", closeWindow)
# close window by keyboard Esc -> trigger event closeWindow
win.bind('<Escape>', (lambda event: closeWindow()))

# Place all components on the screen
scrollbar.place(x=628, y=5, height=453, width=16)
ChatLog.place(x=3, y=2, height=459, width=625)
EntryBox.place(x=3, y=460, height=95, width=643)

win.mainloop()

# time.sleep(10)

if to_update_model_s:
    print ("Start to update the model S!")
    import model_textcnn_lstm_S
    model_textcnn_lstm_S.textCNN_LSTM_model()
    print ("The model S is updated!")

if to_update_model_f:
    print ("Start to update the model F!")
    train_model('intents_F.json', 'word2Vec_model_F.bin', 'TextCNN', 'model_textcnn_lstm_F.h5')
    print ("The model F is updated!")

if to_update_model_p:
    print ("Start to update the model P!")
    train_model('intents_P.json', 'word2Vec_model_P.bin', 'TextCNN', 'model_textcnn_lstm_P.h5')
    print ("The model P is updated!")