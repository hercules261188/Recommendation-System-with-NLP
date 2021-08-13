import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('final_df_new.csv')
df['customer_reviews_sent_tokens'] = df['customer_reviews'].apply(lambda x: x.split('//')[::4])


def sentiment_analyzer(review_col):
    """
    input the review series, output a new series named 'summary_sentiment' in where 'positive / neutral / negative'
    """
    summary_sentiment = []
    sia = SentimentIntensityAnalyzer()
    row = 0
    for product in review_col:
        for sentence in product:
            sum_compound = 0
            # print(sentence)
            ss = sia.polarity_scores(sentence)
            sum_compound += ss['compound']
            # print(ss)
        avg_compound = sum_compound/(len(review_col[row]))

        # print(f"\n\n{df['product_name'][row]}\navg_compound_score: {round(avg_compound,2)}")
        if avg_compound >= 0.05:
            summary_sentiment.append('positive')
        elif (avg_compound > -0.05) and (avg_compound < 0.05):
            summary_sentiment.append('neutral')
        else:
            summary_sentiment.append('negative')
        row += 1
    return summary_sentiment

df['summary_sentiment'] = sentiment_analyzer(df['customer_reviews_sent_tokens'])

df.to_csv('final_include_NLP_df_13aug0539.csv')