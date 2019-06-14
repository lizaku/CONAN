import csv
import re
import heapq
from response_analysis import isEnglish, remove_emoji
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
import numpy as np
import gensim, logging
import zipfile
from nltk import pos_tag, word_tokenize
from string import punctuation
import random


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

penn2upos = {'NN': 'NOUN',
             'NNP': 'PROPN',
             'NNS': 'NOUN',
             'NNPS': 'PROPN',
             'RB': 'ADV',
             'RBR': 'ADV',
             'VB': 'VERB',
             'VBD': 'VERB', 
             'VBG': 'VERB',
             'VBN': 'VERB',
             'VBZ': 'VERB',
             'VBP': 'VERB',
             'PRP': 'PRON',
             'MD': 'VERB',
             'IN': 'ADP',
             'JJ': 'ADJ',
             'JJR': 'ADJ',
             'JJS': 'ADJ',
             'CC': 'CCONJ',
             'DT': 'DET',
             'TO': 'PART',
             'WP': 'PRON'}
             
cntypes = set()
types_dic = {}

def load_responses_unique_hs():
    with open('../data/final_hate_dataset_train.csv') as f_in:
        data = csv.reader(f_in, delimiter='\t')
        header = next(data)
        train = {}
        for row in data:
            hs = row[1]
            hstype = row[2]
            hssubtype = row[3]
            cn = row[4]
            cntype = row[5]
            cntypes.add(cntype)
            types_dic[cn] = cntype
            train[hs] = (cn, cntype)
    with open('../data/final_hate_dataset_test.csv') as f_in:
        data = csv.reader(f_in, delimiter='\t')
        header = next(data)
        test = {}
        for row in data:
            hs = row[1]
            hstype = row[2]
            hssubtype = row[3]
            cn = row[4]
            cntype = row[5]
            cntypes.add(cntype)
            types_dic[cn] = cntype
            test[hs] = (cn, cntype)
    return train, test
    
def load_responses_all_hs():
    with open('../data/final_hate_dataset_train.csv') as f_in:
        data = csv.reader(f_in, delimiter='\t')
        header = next(data)
        train = {}
        for row in data:
            hs = row[1]
            hstype = row[2]
            hssubtype = row[3]
            cn = row[4]
            cntype = row[5]
            cntypes.add(cntype)
            types_dic[cn] = cntype
            train[(hs, cn)] = [cn, cntype]
            #if hs in train:
            #    train[hs][0] = train[hs][0] + ' ' + response
            #else:
            #    train[hs] = [response, cntype]
    with open('../data/final_hate_dataset_test.csv') as f_in:
        data = csv.reader(f_in, delimiter='\t')
        header = next(data)
        test = {}
        for row in data:
            hs = row[1]
            hstype = row[2]
            hssubtype = row[3]
            cn = row[4]
            cntype = row[5]
            cntypes.add(cntype)
            types_dic[cn] = cntype
            test[(hs, cn)] = (cn, cntype)
            #if hs in test:
            #    test[hs][0] = test[hs][0] + ' ' + response
            #else:
            #    test[hs] = [response, cntype]
    return train, test


def cleaned(s):
    # remove mentions
    s = re.sub('@[^ ]+ ', '', s, flags=re.U|re.I)
    # filter out non-English responses
    if not isEnglish(s):
        s = ''
    # remove emojis
    s = remove_emoji(s)
    # remove URLs
    s = re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    return s
    

def transform_embeddings(test, model):
    h_vecs = []
    r_vecs = []
    h_r_vecs = []
    h_r_matrix = []
    i = 0
    decode = {}
    for h in test:
        decode[i] = (h[0], test[h][0])
        h_tokens = word_tokenize(h[0].lower())
        h_tags = pos_tag(h_tokens)
        r_tokens = word_tokenize(test[h][0].lower())
        r_tags = pos_tag(r_tokens)
        h_vec = []
        r_vec = []
        h_r_vec = []
        # for word in h_tokens:
            # if word in model:
                # h_vec.append(model.get_vector(word))
                # h_r_vec.append(model.get_vector(word))
            # else:
                # pass
        # for word in r_tokens:
            # if word in model:
                # r_vec.append(model.get_vector(word))
                # h_r_vec.append(model.get_vector(word))
            # else:
                # pass
        for w in h_tags:
            if w[1] in punctuation:
                continue
            try:
                word = w[0] + '_' + penn2upos[w[1]]
            except KeyError:
                word = w[0] + '_X'
            if word in model:
                h_vec.append(model.get_vector(word))
                h_r_vec.append(model.get_vector(word))
            else:
                pass
        for w in r_tags:
            try:
                word = w[0] + '_' + penn2upos[w[1]]
            except KeyError:
                word = w[0] + '_X'
            if word in model:
                r_vec.append(model.get_vector(word))
                h_r_vec.append(model.get_vector(word))
            else:
                pass
        h_vec = np.mean(h_vec, axis=0)
        r_vec = np.mean(r_vec, axis=0)
        h_r_vec = np.mean(h_r_vec, axis=0)
        if h_r_matrix == []:
            h_r_matrix = h_r_vec
        else:
            try:
                h_r_matrix = np.vstack((h_r_matrix, h_r_vec))
            except:
                continue
        h_vecs.append(h_vec)
        r_vecs.append(r_vec)
        h_r_vecs.append(h_r_vec)
        i += 1
    i = np.shape(h_r_vecs)[0]
    y1 = np.ones(i)
    if len(list(test.keys())) > 3000:
        neg_sample = random.sample(list(test.keys()), 3000)
    else:
        neg_sample = []
    z = 0
    for h in neg_sample:
        h_tokens = word_tokenize(h[0].lower())
        h_tags = pos_tag(h_tokens)
        true_r = h[1]
        false_r = true_r
        while false_r == true_r:
            false_r = random.choice(list(test.keys()))[1]
        r_tokens = word_tokenize(false_r.lower())
        r_tags = pos_tag(r_tokens)
        h_vec = []
        r_vec = []
        h_r_vec = []
        # for word in h_tokens:
            # if word in model:
                # h_vec.append(model.get_vector(word))
                # h_r_vec.append(model.get_vector(word))
            # else:
                # pass
        # for word in r_tokens:
            # if word in model:
                # r_vec.append(model.get_vector(word))
                # h_r_vec.append(model.get_vector(word))
            # else:
                # pass
        for w in h_tags:
            if w[1] in punctuation:
                continue
            try:
                word = w[0] + '_' + penn2upos[w[1]]
            except KeyError:
                word = w[0] + '_X'
            if word in model:
                h_vec.append(model.get_vector(word))
                h_r_vec.append(model.get_vector(word))
            else:
                pass
        for w in r_tags:
            try:
                word = w[0] + '_' + penn2upos[w[1]]
            except KeyError:
                word = w[0] + '_X'
            if word in model:
                r_vec.append(model.get_vector(word))
                h_r_vec.append(model.get_vector(word))
            else:
                pass
        h_vec = np.mean(h_vec, axis=0)
        r_vec = np.mean(r_vec, axis=0)
        h_r_vec = np.mean(h_r_vec, axis=0)
        if h_r_matrix == []:
            h_r_matrix = h_r_vec
        else:
            try:
                h_r_matrix = np.vstack((h_r_matrix, h_r_vec))
            except:
                continue
        h_vecs.append(h_vec)
        r_vecs.append(r_vec)
        h_r_vecs.append(h_r_vec)
    #h_matrix = np.vstack(h_vecs)
    #r_matrix = np.vstack(r_vecs)
    #h_r_matrix = np.vstack(h_r_vecs)
    y2 = np.zeros(np.shape(h_r_vecs)[0] - i)
    print('y', np.shape(y1)[0], np.shape(y2)[0])
    y = np.concatenate([y1, y2])
    return h_r_matrix, y, h_vecs, r_vecs, decode


def evaluate_embeddings_svm(topn=3):
    with zipfile.ZipFile('../data/enwiki_model.zip', 'r') as archive:
        stream = archive.open('model.bin')
        model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove2word2vec(glove_input_file="../data/glove.6B.50d.txt", word2vec_output_file="../data/gensim_glove.6B.50d.txt")
    # model = gensim.models.KeyedVectors.load_word2vec_format("../data/gensim_glove.6B.50d.txt", binary=False)
    train, test = load_responses_all_hs()
    train = {(cleaned(h[0]) + ' ' + r[1], cleaned(h[1])): (cleaned(r[0]), r[1]) for h, r in train.items() if len(cleaned(r[0])) and len(cleaned(h[0])) > 0}
    test = {(cleaned(h[0]) + ' ' + r[1], cleaned(h[1])): (cleaned(r[0]), r[1]) for h, r in test.items() if len(cleaned(r[0])) and len(cleaned(h[0])) > 0}
    test_decode = {x[0]: x[1] for x in test}
    X_train, y_train, _, _, _ = transform_embeddings(train, model)
    X_test, y_test, h_vecs, r_vecs, decode = transform_embeddings(test, model)
    SVM = svm.SVC()
    SVM.probability=True
    SVM.fit(X_train, y_train)
    guess = 0
    for h in range(len(h_vecs)):
        hs = decode[h][0]
        true_cn = test_decode[hs]
        scores = []
        for r in range(len(r_vecs)):
            h_r_vec = np.mean((h_vecs[h], r_vecs[r]), axis=0)
            score = SVM.predict_proba(h_r_vec.reshape(1, -1))
            true_score = score[0][1]
            scores.append((decode[r][1], true_score))
        scores = sorted(scores, key=lambda x: x[1])
        pred_cn = set([x[0] for x in scores[:topn]])
        if true_cn in pred_cn:
            guess += 1
    print('Accuracy', guess/len(h_vecs), 'guessed', guess, 'out of', len(h_vecs))
    

def evaluate_embeddings_sim(topn=3):
    with zipfile.ZipFile('../data/enwiki_model.zip', 'r') as archive:
        stream = archive.open('model.bin')
        model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    #from gensim.scripts.glove2word2vec import glove2word2vec
    #glove2word2vec(glove_input_file="../data/glove.6B.50d.txt", word2vec_output_file="../data/gensim_glove.6B.50d.txt")
    #model = gensim.models.KeyedVectors.load_word2vec_format("../data/gensim_glove.6B.50d.txt", binary=False)
    train, test = load_responses_all_hs()
    train = {(cleaned(h[0]) + ' ' + r[1], cleaned(h[1])): (cleaned(r[0]), r[1]) for h, r in train.items() if len(cleaned(r[0])) and len(cleaned(h[0])) > 0}
    test = {(cleaned(h[0]) + ' ' + r[1], cleaned(h[1])): (cleaned(r[0]), r[1]) for h, r in test.items() if len(cleaned(r[0])) and len(cleaned(h[0])) > 0}
    test_decode = {x[0]: x[1] for x in test}
    #X_train, y_train, _, _, _ = transform_embeddings(train, model)
    X_test, y_test, h_vecs, r_vecs, decode = transform_embeddings(test, model)
    guess = 0
    for h in range(len(h_vecs)):
        hs = decode[h][0]
        true_cn = test_decode[hs]
        scores = []
        for r in range(len(r_vecs)):
            sim = cosine_similarity(h_vecs[h].reshape(1, -1), r_vecs[r].reshape(1, -1))
            scores.append((decode[r][1], sim))
        scores = sorted(scores, key=lambda x: x[1])
        pred_cn = set([x[0] for x in scores[:topn]])
        if true_cn in pred_cn:
            guess += 1
    print('Accuracy', guess/len(h_vecs), 'guessed', guess, 'out of', len(h_vecs))
    

def evaluate_tfidf(n=3):
    train, test = load_responses_all_hs()
    train = {(cleaned(h[0]), cleaned(h[1])): (cleaned(r[0]), r[1]) for h, r in train.items() if len(cleaned(r[0])) and len(cleaned(h[0])) > 0}
    test = {(cleaned(h[0]), cleaned(h[1])): (cleaned(r[0]), r[1]) for h, r in test.items() if len(cleaned(r[0])) and len(cleaned(h[0])) > 0}
    tfidf_vectorizer = TfidfVectorizer()
    print(len(train))
    # Adding argument types to the HS
    #train_hs_cn = [h[0] + ' ' + train[h][1] + ' ' + train[h][0] for h in train]
    # Without adding argument types to the HS
    train_hs_cn = [h[0] + ' ' + train[h][0] for h in train]
    test_cn = [test[x][0] for x in test]
    tfidf_vectorizer.fit(train_hs_cn)
    test_cn_matrix = tfidf_vectorizer.transform(test_cn)
    guess = 0
    for hs in test:
        # Adding argument types to the HS
        #vec = tfidf_vectorizer.transform([hs[0] + ' ' + test[hs][1]])
        # Without adding argument types to the HS
        vec = tfidf_vectorizer.transform([hs[0]])
        res = cosine_similarity(vec, test_cn_matrix)[0]
        n_largest = heapq.nlargest(n, range(len(res)), res.take)
        id_out = np.argmax(res)
        pred_cs = [test_cn[i] for i in n_largest]
        if test[hs][0] in pred_cs:
            guess += 1
        print('hate speech:\n', hs[0])
        print('predicted response:\n', pred_cs)
        print('true response:\n', test[hs][0])
        print('\n')
    
    print('Accuracy', guess/len(test), ', guessed', guess, ', overall', len(test))

if __name__ == '__main__':
    #evaluate_tfidf(10)
    evaluate_embeddings_svm(10)
    #evaluate_embeddings_sim(10)
