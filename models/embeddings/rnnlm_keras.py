import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlflow import log_metric, log_param

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path

# HYPERPARAMETERS BEGIN ###############################################
MAX_ARTICLE_LENGTH = 1000
EMBEDDING_VECTOR_LENGTH = 50
EMBEDDING_VOCAB_SIZE = 400000
LSTM_MEMORY_SIZE = 100
NN_OPTIMIZER = 'adam'
NN_LOSS_FUNCTION = 'binary_crossentropy'
NN_EPOCHS = 1
USE_GLOVE_EMBEDDINGS = False
NN_BATCH_SIZE = 128
# HYPERPARAMETERS END #################################################

# Other config parameters
RANDOM_SEED = 42
GLOVE_FILEPATH = 'glove.6B.%dd.txt' % EMBEDDING_VECTOR_LENGTH
FR_DATASET_PATH = "../../data/fakerealnews_GeorgeMcIntire/fake_or_real_news.csv"
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


# Please add a function like this to read any other dataset
def read_mcintire_dataset():
    print("Reading dataset")
    fr = pd.read_csv(FR_DATASET_PATH)
    print("Read dataset")
    fr = fr.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    fr['title_and_text'] = fr['title'] + ' ' + fr['text']
    model, embedding_matrix = load_glove_model_v2(EMBEDDING_VECTOR_LENGTH)
    fr['title_and_text_cleaned'] = fr['title_and_text'].apply(lambda a: cleanArticle(a))
    fr['news_embed_idx'] = fr['title_and_text_cleaned'].apply(lambda a: article_to_word_id_list(a, model))

    X_train, X_test, y_train, y_test = \
        train_test_split(fr['news_embed_idx'], np.where(fr['label'] == 'FAKE', 1, 0),
                         test_size=.2, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, embedding_matrix


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    X_train, X_test, y_train, y_test, embedding_matrix = read_mcintire_dataset()

    # Add padding if needed
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_ARTICLE_LENGTH)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_ARTICLE_LENGTH)

    # Define model
    model = Sequential()
    if USE_GLOVE_EMBEDDINGS:
        model.add(Embedding(EMBEDDING_VOCAB_SIZE, EMBEDDING_VECTOR_LENGTH, weights=[embedding_matrix],
                            input_length=MAX_ARTICLE_LENGTH, trainable=False))
    else:
        model.add(Embedding(EMBEDDING_VOCAB_SIZE, EMBEDDING_VECTOR_LENGTH, input_length=MAX_ARTICLE_LENGTH))

    # Question: How to decide what initializers to use?
    model.add(LSTM(LSTM_MEMORY_SIZE))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=NN_LOSS_FUNCTION, optimizer=NN_OPTIMIZER, metrics=['accuracy'])
    print(model.summary())

    # Log parameters
    log_param("MAX_ARTICLE_LENGTH", MAX_ARTICLE_LENGTH)
    log_param("USE_GLOVE_EMBEDDINGS", USE_GLOVE_EMBEDDINGS)
    log_param("EMBEDDING_VOCAB_SIZE", EMBEDDING_VOCAB_SIZE)
    log_param("EMBEDDING_VECTOR_LENGTH", EMBEDDING_VECTOR_LENGTH)
    log_param("LSTM_MEMORY_SIZE", LSTM_MEMORY_SIZE)
    log_param("NN_LOSS_FUNCTION", NN_LOSS_FUNCTION)
    log_param("NN_OPTIMIZER", NN_OPTIMIZER)
    log_param("NN_EPOCHS", NN_EPOCHS)
    log_param("NN_BATCH_SIZE", NN_BATCH_SIZE)

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE)

    # Predict model
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy = scores[1] * 100.0
    log_metric("Accuracy", accuracy)
    print("Accuracy: %.2f%%" % accuracy)