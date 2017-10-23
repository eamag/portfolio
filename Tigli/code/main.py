import pandas as pd
from gensim import corpora, models, similarities
from nltk.stem.wordnet import WordNetLemmatizer
from stop_words import get_stop_words
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
# import nltk; nltk.download(['wordnet', 'stopwords'])


def load_data():
    df = pd.read_csv('../data/realDonaldTrump_tweets.csv')
    df['hour'] = pd.to_datetime(df['created_at']).dt.hour  # TODO: use as feature
    df.drop(['id', 'created_at'], axis=1, inplace=True)
    df['text'] = df['text'].str[2:-1]
    return df.text.values.tolist()


def tokens_from_word(word):
    tokenizer = TweetTokenizer(preserve_case=False)
    tokens = tokenizer.tokenize(word)
    en_stop = get_stop_words('en')
    stemmer = WordNetLemmatizer()
    return [stemmer.lemmatize(i) for i in tokens if i not in en_stop]


def lda_model(doc):
    # TODO: replace mentions and hashtags
    twitts_list = load_data()
    texts = [tokens_from_word(text) for text in twitts_list]
    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    # corpora.MmCorpus.serialize('trump_corpus.mm', corpus)  # store to disk, for later use

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)  # initialize an LSI trans

    word = tokens_from_word(doc)
    vec_bow = dictionary.doc2bow(word)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space

    index = similarities.MatrixSimilarity(lsi[corpus])

    sims = index[vec_lsi]  # perform a similarity query against the corpus
    max_score, min_score = str(sims.max()), str(sims.min())
    if float(max_score) <= 0:
        output = "That doesn't look like D. Trump at all!\n"
    else:
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        # sims_scores = [x[1] for x in sims]
        output = '\nMost similar tweet: ' + twitts_list[sims[0][0]] + '  # max score: '  # + max_score + '\n\n'
        # output += 'least similar: ' + twitts_list[sims[-1][0]] + '  # min score: ' + min_score + '\n\n'
        # output += 'original: ' + doc + '\n\n'
    return output, max_score


def oneclasssvm(doc):
    twitts_list = load_data()
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit(twitts_list)
    text_transformed = vectorizer.transform(twitts_list)
    clf = OneClassSVM(random_state=0)
    clf.fit(text_transformed)
    lis = []
    lis.insert(0, doc)
    doc_tf = tf_idf.transform(lis)
    return clf.predict(doc_tf)[0]


if __name__ == '__main__':
    doc = 'Despite what you hear in the press, healthcare is coming along great. We are talking ' \
          'to many groups and it will end in a beautiful picture! '
    doc2 = 'LinkedIn Workforce Report: January and February were the strongest consecutive' \
           ' months for hiring since August 2 and September 2015 '
    print(lda_model(doc))
    print(oneclasssvm(doc))
