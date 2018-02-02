import sys
import os
import cPickle
sys.path.append('../src')
import data_io, params, SIF_embedding
import nltk
import pdb
import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy.io import loadmat

DATA_DIR = '/home/shunan/Code/Data/'

def generate_imdb_sif_embeddings(imdb_data, word_vectors, words, weight_param):
    '''
    Generate the SIF embeddings for the IMDB data.
    '''

    print('Generating SIF embeddings for weight value: {}'.format(weight_param))

    # Getting the IMDB data
    train_data = imdb_data['train_data']
    test_data = imdb_data['test_data']
    train_labels = imdb_data['train_labels']
    I = train_labels != 0
    train_data = train_data[I]

    # preprocessing
    train_sentences = []
    test_sentences = []
    for i in range(train_data.shape[0]):
        tokens = nltk.word_tokenize(train_data[i][0].strip().lower())
        sen = []
        for tok in tokens:
            if tok in words:
                sen.append(tok)
        train_sentences.append(' '.join(sen))

    for i in range(test_data.shape[0]):
        tokens = nltk.word_tokenize(test_data[i][0][0].strip().lower())
        sen = []
        for tok in tokens:
            if tok in words:
                sen.append(tok)
        test_sentences.append(' '.join(sen))

    # Setup for computing SIF embeddings.
    model_params = params.params()
    model_params.rmpc = 1

    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
    word2weight = data_io.getWordWeight(weightfile, weight_param)
    weight4ind = data_io.getWeight(words, word2weight)
    train_x, train_m = data_io.sentences2idx(train_sentences, words)
    train_w = data_io.seq2weight(train_x, train_m, weight4ind)
    train_embedding = SIF_embedding.SIF_embedding(word_vectors, train_x, train_w, model_params)
    np.save(os.path.join(DATA_DIR, 'SIF/train_embeddings_{}.npy'.format(weight_param)), train_embedding)

    test_x, test_m = data_io.sentences2idx(test_sentences, words)
    test_w = data_io.seq2weight(test_x, test_m, weight4ind)
    test_embedding = SIF_embedding.SIF_embedding(word_vectors, test_x, test_w, model_params)
    np.save(os.path.join(DATA_DIR, 'SIF/test_embeddings_{}.npy'.format(weight_param)), test_embedding)


def generate_amazon_sif_embeddings(train_data, test_data, word_vectors, words, weight_param):
    '''
    Generate the SIF embeddings for the Amazon data.
    '''

    print('Generating SIF embeddings for weight value: {}'.format(weight_param))

    # preprocessing
    train_sentences = []
    test_sentences = []

    tokenizer = TweetTokenizer()

    for sen in train_data:
        sen = sen.strip().lower()
        tmp = nltk.word_tokenize(' '.join(tokenizer.tokenize(sen)))
        new_sen = [tok for tok in tmp if tok in words]
        train_sentences.append(' '.join(new_sen))

    for sen in test_data:
        sen = sen.strip().lower()
        tmp = nltk.word_tokenize(' '.join(tokenizer.tokenize(sen)))
        new_sen = [tok for tok in tmp if tok in words]
        test_sentences.append(' '.join(new_sen))

    # Setup for computing SIF embeddings.
    model_params = params.params()
    model_params.rmpc = 1

    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
    word2weight = data_io.getWordWeight(weightfile, weight_param)
    weight4ind = data_io.getWeight(words, word2weight)

    train_x, train_m = data_io.sentences2idx(train_sentences, words)
    train_w = data_io.seq2weight(train_x, train_m, weight4ind)
    train_embedding = SIF_embedding.SIF_embedding(word_vectors, train_x, train_w, model_params)
    np.save(os.path.join(DATA_DIR, 'amazon_food/SIF/train_embeddings_{}.npy'.format(weight_param)), train_embedding)

    test_x, test_m = data_io.sentences2idx(test_sentences, words)
    test_w = data_io.seq2weight(test_x, test_m, weight4ind)
    test_embedding = SIF_embedding.SIF_embedding(word_vectors, test_x, test_w, model_params)
    np.save(os.path.join(DATA_DIR, 'amazon_food/SIF/test_embeddings_{}.npy'.format(weight_param)), test_embedding)

if __name__ == '__main__':

    IMDB = False

    # Read word2vec dictionary
    word_vectors = loadmat(os.path.join(DATA_DIR, 'word2vec/GoogleNews-vectors-negative300.mat'))
    word_vectors = word_vectors['vectors']

    words = dict()
    f = open(os.path.join(DATA_DIR, 'word2vec/dict.txt'), 'r')
    i = 0
    word = f.readline()
    while word != '':
        words[word.strip()] = i
        i += 1
        word = f.readline()
    f.close()

    if IMDB:
        imdb_data = loadmat(os.path.join(DATA_DIR, 'imdb_sentiment/imdb_sentiment.mat'))

        param_values = [0.001, 0.005, 0.01, 0.1]
        for p in param_values:
            generate_imdb_sif_embeddings(imdb_data, word_vectors, words, p)
    else:
        with open(os.path.join(DATA_DIR, 'amazon_food/train_data.pkl'), 'r') as f:
            train_data = cPickle.load(f)
            train_data = train_data[0]

        with open(os.path.join(DATA_DIR, 'amazon_food/test_data.pkl'), 'r') as f:
            test_data = cPickle.load(f)
            test_data = test_data[0]

        param_values = [0.001, 0.005, 0.01, 0.1]
        for p in param_values:
            generate_amazon_sif_embeddings(train_data, test_data, word_vectors, words, p)
