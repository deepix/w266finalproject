import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path

# HYPERPARAMETERS BEGIN ###############################################
RANDOM_SEED = 42
MAX_ARTICLE_LENGTH = 1000
EMBEDDING_VECTOR_LENGTH = 50
EMBEDDING_VOCAB_SIZE = 400000
LSTM_MEMORY_SIZE = 100
NN_OPTIMIZER = 'adam'
NN_LOSS_FUNCTION = 'binary_crossentropy'
NN_EPOCHS = 3
NN_BATCH_SIZE = 128
# HYPERPARAMETERS END #################################################

# Other config parameters
GLOVE_FILEPATH = 'glove.6B.50d.txt'
DATASET_PATH = "../../data/fakerealnews_GeorgeMcIntire/fake_or_real_news.csv"
ID_UNKNOWN = 399999


def cleanArticle(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def load_glove_model_v2(dim=50):
    """Load a Glove model into a gensim model, converting it
    into word2vec if necessary.
    Adapted from: https://stackoverflow.com/a/47465278
    """
    print("Loading Glove embedding")
    glove_data_file = GLOVE_FILEPATH
    word2vec_output_file = '%s.w2v' % glove_data_file

    if not Path(word2vec_output_file).exists():
        glove2word2vec(glove_input_file=glove_data_file, word2vec_output_file=word2vec_output_file)
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print("Loaded Glove embedding")
    return glove_model


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
    fr = pd.read_csv(DATASET_PATH)
    print("Read dataset")
    fr = fr.sample(frac=1, random_state=42).reset_index(drop=True)

    fr['title_and_text'] = fr['title'] + ' ' + fr['text']
    fr['title_and_text_cleaned'] = fr['title_and_text'].apply(cleanArticle)

    model = load_glove_model_v2(EMBEDDING_VECTOR_LENGTH)

    fr['news_embed_idx'] = fr['title_and_text_cleaned'].apply(lambda x: article_to_word_id_list(x, model))

    X_train, X_test, y_train, y_test = \
        train_test_split(fr['news_embed_idx'], np.where(fr['label'] == 'FAKE', 1, 0),
                         test_size=.2, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    X_train, X_test, y_train, y_test = read_mcintire_dataset()

    # Add padding if needed
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_ARTICLE_LENGTH)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_ARTICLE_LENGTH)

    # Define model
    model = Sequential()
    model.add(Embedding(EMBEDDING_VOCAB_SIZE, EMBEDDING_VECTOR_LENGTH, input_length=MAX_ARTICLE_LENGTH))
    model.add(LSTM(LSTM_MEMORY_SIZE))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=NN_LOSS_FUNCTION, optimizer=NN_OPTIMIZER, metrics=['accuracy'])
    print(model.summary())

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE)

    # Predict model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
