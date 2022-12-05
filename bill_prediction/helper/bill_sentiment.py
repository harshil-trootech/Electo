from bill.models import *
from bill_prediction.constants import *
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import random
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class BillSentiment:
    # def __init__(self):
    positive_status_list = ['ENACTED:SIGNED']
    negative_status_list = ['FAIL:ORIGINATING:HOUSE', 'FAIL:ORIGINATING:SENATE', 'ENACTED:VETO_OVERRIDE',
                            'FAIL:SECOND:HOUSE', 'FAIL:SECOND:SENATE', 'PASS_BACK:HOUSE', 'PASS_BACK:SENATE']
    vocab = []
    stopwords = stopwords.words('english')
    stemmer = PorterStemmer()

    @classmethod
    def train_sentiment_model(cls):
        sentiment_bills_positive = Bill.objects.filter(status__in=cls.positive_status_list)
        sentiment_bills_negative = Bill.objects.filter(status__in=cls.negative_status_list)
        positive_bills = []
        negative_bills = []
        print("...Processing bill's summary...")
        for bill in sentiment_bills_positive:
            summary = bill.summary
            if summary:
                positive_bills.append([cls.get_clean_summary(summary, training=True), 1])
            else:
                positive_bills.append([bill.official_title.lower(), 1])

        for bill in sentiment_bills_negative:
            summary = bill.summary
            if summary:
                negative_bills.append([cls.get_clean_summary(summary, training=True), 0])
            else:
                negative_bills.append([bill.official_title.lower(), 0])
        cls.vocab = list(set(cls.vocab))
        bills = negative_bills + random.sample(positive_bills, len(negative_bills))
        random.shuffle(bills)

        x = [bill[0] for bill in bills]
        y = [bill[1] for bill in bills]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
        print("...Training model...")
        model = cls.get_model()
        model.fit(x_train, y_train, batch_size=2, epochs=3, validation_data=(x_test, y_test))
        model.save(BILL_SENTIMENT_MODEL_PATH)

    @classmethod
    def get_clean_summary(cls, summary, training=False):
        final_sentences = []
        summary = ')'.join(summary.split(')')[1:])
        sentences = sent_tokenize(summary)
        clean_sentences = [re.sub(r'[^A-Za-z ]', '', sentence.lower()) for sentence in sentences]
        for sent in clean_sentences:
            final_sentences.append(' '.join([word for word in sent.strip().split(' ') if word != '']))
            if training:
                cls.vocab.extend([word for word in sent.strip().split(' ') if word != ''])
        return ' '.join(final_sentences)

    @classmethod
    def get_model(cls):
        model = models.Sequential()
        model.add(layers.TextVectorization(output_sequence_length=64, vocabulary=cls.vocab))
        model.add(layers.Embedding(len(cls.vocab)+2, 8))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(8, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
