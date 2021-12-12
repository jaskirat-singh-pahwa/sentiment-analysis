import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
import gensim.corpora as corpora
from nltk.corpus import stopwords
from pprint import pprint


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def get_topics(file_path):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    reviews = pd.read_csv(file_path)
    reviews['paper_text_processed'] = reviews['review'].map(lambda x: re.sub('[,\.!?]', '', x))
    reviews['paper_text_processed'] = reviews['paper_text_processed'].map(lambda x: x.lower())
    reviews['paper_text_processed'].head()
    # long_string = ','.join(list(reviews['paper_text_processed'].values))
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'br', 'movie', 'film', 'one'])

    data = reviews.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    data_words = remove_stopwords(data_words, stop_words)

    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    num_topics = 10
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics
    )
    pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
