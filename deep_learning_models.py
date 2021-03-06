import re
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlflow import log_metric, log_param

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path

# HYPERPARAMETERS BEGIN ###############################################
# These are set upon call to do_run()
MAX_ARTICLE_LENGTH = None
EMBEDDING_VECTOR_LENGTH = None
EMBEDDING_VOCAB_SIZE = None
LSTM_MEMORY_SIZE = None
NN_OPTIMIZER = None
NN_LOSS_FUNCTION = None
NN_EPOCHS = None
USE_GLOVE_EMBEDDINGS = None
NN_BATCH_SIZE = None
DATASET = None
DROPOUT_RATE = None
NN_ARCH_TYPE = None
# HYPERPARAMETERS END #################################################

# Other config parameters
RANDOM_SEED = 42
FR_DATASET_PATH = "data/fakerealnews_GeorgeMcIntire/fake_or_real_news.csv"
PEREZ_DATASET_PATH = "data/fakeNewsDatasets_Perez-Rosas2018"
ID_UNKNOWN = 399999


def cleanArticle(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def load_glove_model_v2(dim):
    """Load a Glove model into a gensim model, converting it
    into word2vec if necessary.
    Adapted from: https://stackoverflow.com/a/47465278
    """
    GLOVE_FILEPATH = 'models/embeddings/glove.6B.%dd.txt' % EMBEDDING_VECTOR_LENGTH
    print("Loading Glove embedding")
    glove_data_file = GLOVE_FILEPATH
    word2vec_output_file = '%s.w2v' % glove_data_file

    if not Path(word2vec_output_file).exists():
        glove2word2vec(glove_input_file=glove_data_file, word2vec_output_file=word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print("Loaded Glove embedding")

    embedding_matrix = np.zeros((len(model.vocab), dim))
    for i in range(len(model.vocab)):
        embedding_vector = model[model.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return model, embedding_matrix


def article_to_word_id_list(article, model):
    word_index_list = []
    word_list = article.split()
    for i, word in enumerate(word_list):
        if word in model.vocab:
            word_index_list.append(model.vocab[word].index)
        else:
            # Unknown
            word_index_list.append(ID_UNKNOWN)
    return word_index_list


def read_mcintire_dataset():
    print("Reading dataset")
    fr = pd.read_csv(FR_DATASET_PATH)
    fr = fr.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    fr['title_and_text'] = fr['title'] + ' ' + fr['text']
    model, embedding_matrix = load_glove_model_v2(EMBEDDING_VECTOR_LENGTH)
    fr['title_and_text_cleaned'] = fr['title_and_text'].apply(lambda a: cleanArticle(a))
    fr['news_embed_idx'] = fr['title_and_text_cleaned'].apply(lambda a: article_to_word_id_list(a, model))

    X_train, X_test, y_train, y_test = \
        train_test_split(fr['news_embed_idx'], np.where(fr['label'] == 'FAKE', 1, 0),
                         test_size=.2, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)

    print("Finished reading dataset")
    return X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix


def read_perez_dataset(dataset_name):
    
    def remove_numbers(in_str):
        return re.sub(r'[0-9]+', '', in_str)
    
    print("Reading dataset")
    result_data_list = []
    data_dir = PEREZ_DATASET_PATH
    for news_type in ['fake', 'legit']:
        folder = '%s/%s/%s' % (data_dir, dataset_name, news_type)
        for fname in os.listdir(folder):
            result_data = {}
            result_data['dataset_name'] = dataset_name
            result_data['news_type'] = news_type
            if news_type == 'fake':
                result_data['is_fake'] = 1
            else:
                result_data['is_fake'] = 0
            if dataset_name == 'fakeNewsDataset':
                result_data['news_category'] = remove_numbers(fname.split('.')[0])
            result_data['file_name'] = fname
            filepath = os.path.join(folder, fname)
            with open(filepath, 'r', encoding="utf8") as f:
                file_data = f.read().split('\n')
                # Some articles don't have a headline, but only article body.
                if len(file_data) > 1:
                    news_content_data = ' '.join(file_data[2:])
                    result_data['news_headline'] = file_data[0]
                else:
                    news_content_data = file_data[0]
                    result_data['news_headline'] = ''
                result_data['news_content'] = news_content_data
                result_data['news_all'] = ' '.join(file_data[0:])
                result_data_list.append(result_data)
                
    df = pd.DataFrame(result_data_list)
    
    model, embedding_matrix = load_glove_model_v2(EMBEDDING_VECTOR_LENGTH)
    df['news_all_clean'] = df['news_all'].apply(lambda a: cleanArticle(a))
    df['news_embed_idx'] = df['news_all_clean'].apply(lambda a: article_to_word_id_list(a, model))
    
    X_train, X_test, y_train, y_test = train_test_split(df['news_embed_idx'], df['is_fake'], test_size=.2, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)
    
    print("Finished reading dataset")
    return X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix


def set_model_hyperparameters(hyperparameter_dict):
    for k, v in hyperparameter_dict.items():
        globals()[k] = v
        log_param(k, v)


def do_run(hyperparameter_dict):

    set_model_hyperparameters(hyperparameter_dict)

    np.random.seed(RANDOM_SEED)

    X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix = read_perez_dataset(DATASET)
    # X_train, X_val, X_test, y_train, y_val, y_test, embedding_matrix = read_mcintire_dataset()
    
    # Add padding if needed
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_ARTICLE_LENGTH)
    X_val = sequence.pad_sequences(X_val, maxlen=MAX_ARTICLE_LENGTH)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_ARTICLE_LENGTH)

    # Define model
    model = Sequential()
    if USE_GLOVE_EMBEDDINGS:
        model.add(Embedding(EMBEDDING_VOCAB_SIZE, EMBEDDING_VECTOR_LENGTH, weights=[embedding_matrix],
                            input_length=MAX_ARTICLE_LENGTH, trainable=False))
    else:
        model.add(Embedding(EMBEDDING_VOCAB_SIZE, EMBEDDING_VECTOR_LENGTH, input_length=MAX_ARTICLE_LENGTH))

    # Neural network type
    if NN_ARCH_TYPE == '2layerLSTM':
        model.add(LSTM(LSTM_MEMORY_SIZE, dropout=DROPOUT_RATE, return_sequences=True, input_shape=(MAX_ARTICLE_LENGTH, EMBEDDING_VECTOR_LENGTH)))
        #model.add(LSTM(LSTM_MEMORY_SIZE, dropout=DROPOUT_RATE, return_sequences=True))  # Can add this to make 3 layers
        model.add(LSTM(LSTM_MEMORY_SIZE, dropout=DROPOUT_RATE))
    elif NN_ARCH_TYPE == '1layerLSTM':
        model.add(LSTM(LSTM_MEMORY_SIZE, dropout=DROPOUT_RATE))
    elif NN_ARCH_TYPE == '1layerGRU':
        model.add(GRU(LSTM_MEMORY_SIZE, dropout=DROPOUT_RATE))
    else:
        assert False, "Unknown NN arch type"

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=NN_LOSS_FUNCTION, optimizer=NN_OPTIMIZER, metrics=['accuracy'])
    print(model.summary())
    
    # Train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE)
    
    # Predict model on validation (Dev) set 
    scores = model.evaluate(X_val, y_val, verbose=1)
    accuracy = scores[1] * 100
    log_metric('accuracy', accuracy)
    print("Accuracy: %.2f%%" % accuracy)
    
    # Predict model on test set 
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy = scores[1] * 100
    log_metric('accuracy', accuracy)
    print("Accuracy on Test Set: %.2f%%" % accuracy)
    
    # Confusion matrix of results (ensure it doesn't predict the same class for all records)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    hyperparameter_dict_list = [
        {
            'MAX_ARTICLE_LENGTH': 500,
            'EMBEDDING_VECTOR_LENGTH': 50,
            'EMBEDDING_VOCAB_SIZE': 400000,
            'LSTM_MEMORY_SIZE': 100,
            'NN_OPTIMIZER': optimizers.Adam(lr=0.0001),
            'NN_LOSS_FUNCTION': 'binary_crossentropy',
            'NN_EPOCHS': 35,
            'USE_GLOVE_EMBEDDINGS': False,
            'NN_BATCH_SIZE': 50,
            'DATASET': 'celebrityDataset',
            'DROPOUT_RATE': 0.5,
            'NN_ARCH_TYPE': '1layerLSTM',
        },
        {
            'MAX_ARTICLE_LENGTH': 500,
            'EMBEDDING_VECTOR_LENGTH': 50,
            'EMBEDDING_VOCAB_SIZE': 400000,
            'LSTM_MEMORY_SIZE': 100,
            'NN_OPTIMIZER': optimizers.Adam(lr=0.0001),
            'NN_LOSS_FUNCTION': 'binary_crossentropy',
            'NN_EPOCHS': 25,
            'USE_GLOVE_EMBEDDINGS': False,
            'NN_BATCH_SIZE': 50,
            'DATASET': 'celebrityDataset',
            'DROPOUT_RATE': 0.5,
            'NN_ARCH_TYPE': '2layerLSTM',
        },
        {
            'MAX_ARTICLE_LENGTH': 500,
            'EMBEDDING_VECTOR_LENGTH': 50,
            'EMBEDDING_VOCAB_SIZE': 400000,
            'LSTM_MEMORY_SIZE': 100,
            'NN_OPTIMIZER': optimizers.Adam(lr=0.0001),
            'NN_LOSS_FUNCTION': 'binary_crossentropy',
            'NN_EPOCHS': 45,
            'USE_GLOVE_EMBEDDINGS': False,
            'NN_BATCH_SIZE': 50,
            'DATASET': 'celebrityDataset',
            'DROPOUT_RATE': 0.5,
            'NN_ARCH_TYPE': '1layerGRU',
        },
    ]

    for hd in hyperparameter_dict_list:
        do_run(hd)

    # Other runs go below
