# ShakespearePlayGenerator

The code for training the models and showing the test resutls can be found in `train.py`.

W2V word vectors are made by `utils/w2vModel.py` script.

## Abstract
Efficient text summarization systems are becoming increasingly necessary as digital content grows at
an exponential rate. These systems can revolutionize various fields such as academic research and news
by compressing lengthy texts into small briefs. The advent of advanced machine learning models in
Natural Language Processing (NLP) and the transformers facilitates the development of such systems
and makes them more feasible than ever.

In this project, we delve into the potential of abstractive text summarization, which aims to generate summaries similar to how a human would - by understanding the essence of the text and producing
a concise, coherent summary in new words. Compared to extractive summarization, which chooses
only the primary sentences from original text, abstractive summarization provides contextually rich
summaries that are much more meaningful and contextually relevant, though imposing additional complexity. By leveraging recent advancements in pre-trained transformer models, more precisely T5[5]
(Text-To-Text Transfer Transformer) and GPT-2 [4](Generative Pretrained Transformer 2), our approach consists of using self-attention mechanisms to comprehend the context of the sequence of words
in a text. Unlike traditional architectures such as Recurrent Neural Networks (RNNs) and Long-Short
Term Memory (LSTMs) networks, the self-attention mechanism makes them exceptionally suited for
abstractive summarization.

Throughout the scope of this work, we harnessed the general language understanding abilities of
pre-trained models of textual data, and their potential to accomplish an abstractive summarization.
To adapt these models to our task, we fine-tuned T5 and GPT2 transformers on a subset of the XSUM
dataset [3], which is an extensive compilation of BBC articles along with their short summaries specifically crafted for summarization tasks.

To assess the performance of our models, we used established evaluation metrics like ROUGE [2]
(Recall-Oriented Understudy for Gisting Evaluation) for a quantitative measure of their quality and
Bert Scores [6] leveraging contextualized embeddings from BERT (Bidirectional Encoder Representations from Transformers) to measure the closeness of generated summaries. Furthermore, We also
supplement this with a subjective evaluation of the generated summaries to compare the two models’
performance.
