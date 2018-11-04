## Detecting Fake News with Natural Language Processing

##### Final Project for MIDS W266: Natural Language Processing with Deep Learning

- Zach Merritt
- Deepak Nagaraj
- Mike Powers

Next Steps:

1) Implement a LSTM model that results in classifcation of "fake" or "legit". We should think about using word2vec or glove for our word embeddings rather than training our own, since we are not using a ton of training data. Train and test this using the Perez-Rosas data.

2) Implement a Naive Bayes model. This will be a more simple model. It will be useful to verify that the LSTM out-performs it.

3) If we have time, train the model on the merged Perez-Rosas AND GeorgeMcIntire data, and then test on the Perez-Rosas to see if we get performance improvements. It is likely that this will not result in improvement, but we can discuss how this model could be better to used for future analyses because it is trained on more data and may be able to generalize better.