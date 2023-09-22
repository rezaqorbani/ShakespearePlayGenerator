# ShakespearePlayGenerator

The code for training the models and showing the test resutls can be found in `train.py`.

W2V word vectors are made by `utils/w2vModel.py` script.

## Abstract
In this report, we investigate the effectiveness of Recurrent Neural Networks (RNNs)
and Long Short-Term Memory (LSTM) models in emulating the unique literary style of
William Shakespeare. Our models are trained on a corpus of Shakespeare’s plays, with the
objective of generating text resembling his stylistic nuances. We analyze the performance
of different models and their variations, considering factors such as the use of Byte-Pair
Encoding (BPE) tokenization, Word2Vec embeddings, and data augmentation techniques.
We further conduct a grid search for optimal learning rate and batch size, and examine the
influence of varying the number of hidden nodes. Nucleus and Temperature Sampling are
employed to balance the novelty and coherence of the generated text. Our experiments reveal
intriguing differences in the capabilities of each model, with LSTM demonstrating superior
performance in certain scenarios. While data augmentation does not significantly enhance
model performance, it has potential in aiding model generalization. The detailed analysis of
these findings and the resultant models’ text-generation abilities will be elaborated upon in
the main body of this report
