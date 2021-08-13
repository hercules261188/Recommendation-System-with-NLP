import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
nltk.download('punkt')
nltk.download('stopwords')


# import the raw csv file
df = pd.read_csv('amazon_co-ecommerce_sample.csv')


def check_na(df):
    """ 
    check number of na values; and number of unique values in each column in the raw data set 
    """
    col_dict = {}
    for col in df.columns:
        col_dict[col] = [df[col].isnull().sum()]
        col_dict[col].append(round((df[col].isnull().sum())/df.shape[0] * 100, 4))
        col_dict[col].append(df[col].nunique())
    df_na = pd.DataFrame(data=col_dict.values(), index=col_dict.keys(), columns=['na_num', 'na_percent', 'num_of_unique_values'])
    df_na = df_na.sort_values(by=['na_num', 'na_percent', 'num_of_unique_values'], ascending=False)
    return df_na


def price_tag_converter(df):
    """ 
    convert the column 'price' to new columns named 'lower_price' , 'upper_price' , 'avg_price' , 'price_tag' 
    """
    df['lower_price'] = df['price'].apply(lambda x: re.findall('(\d\S+)',x)[0].replace(',','') if type(x)==str else np.nan).astype(float)
    df['upper_price'] = df['price'].apply(lambda x: re.findall('(\d\S+)',x)[-1].replace(',','') if type(x)==str else np.nan).astype(float)
    df['avg_price'] = [(i+j)/2 if i>0 else 0 for i,j in zip(df['lower_price'], df['upper_price'])]
    
    price_tag = []
    for p in df['avg_price']:
        if p == 0:
            price_tag.append('no_price')
        elif (p > 0 and p <= 25):
            price_tag.append('below_or_equal_twentyfive')
        elif (p > 25 and p <= 50):
            price_tag.append('twentyfive_to_fifty')
        elif (p > 50 and p <= 100):
            price_tag.append('fifty_to_hundred')
        elif (p > 100 and p <= 200):
            price_tag.append('one_to_two_hundred')
        else:
            price_tag.append('two_hundred_above')
    df['price_tag'] = price_tag
    return df


def clean_some_cols(df):
    # extract and convert to the number value from the string value in column 'number_available_in_stock'
    df['number_available_in_stock'] = df['number_available_in_stock'].apply(lambda x: ''.join(re.findall('(\d+)', x)) if type(x)==str else 0).astype(int)

    # convert column 'number_of_reviews' to int type values, for those with Nan converted to 0
    df['number_of_reviews'] = df['number_of_reviews'].apply(lambda x: re.findall('(\d.*)', x)[0].replace(',','') if type(x)==str else 0).astype('int')

    # convert column 'number_of_answered_questions' to int type values, for those with Nan converted to 0
    df['number_of_answered_questions'] = df['number_of_answered_questions'].apply(lambda x: int(x) if str(x)!='nan' else 0).astype('int')

    # convert column 'average_review_rating' to int type values, for those with Nan converted to 0
    df['average_review_rating'] = df['average_review_rating'].apply(lambda x: float(re.findall('(\S+)', x)[0]) if type(x)==str else x).astype('float')

    # convert column 'amazon_category_and_sub_category' to string format with separated keywords, for those with Nan converted to 'uncategorized'
    df['amazon_category_and_sub_category'] = df['amazon_category_and_sub_category'].apply(lambda x: ','.join(x.split('>')) if type(x)==str else 'uncategorized')

    # convert column 'customers_who_bought_this_item_also_bought' to string format with separated keywords, for those with Nan converted to 'unlinked'
    df['customers_who_bought_this_item_also_bought'] = df['customers_who_bought_this_item_also_bought'].apply(lambda x: ','.join(x.split('|')) if type(x)==str else 'unlinked')

    # convert column 'items_customers_buy_after_viewing_this_item' to string format with separated keywords, for those with Nan converted to 'unlinked'
    df['items_customers_buy_after_viewing_this_item'] = df['items_customers_buy_after_viewing_this_item'].apply(lambda x: ','.join(x.split('|')) if type(x)==str else 'unlinked')

    # convert column 'customer_questions_and_answers' to string format with separated keywords, for those with Nan converted to 'no_q_and_a'
    df['customer_questions_and_answers'] = df['customer_questions_and_answers'].apply(lambda x: ','.join(x.split('//')) if type(x)==str else 'no_q_and_a')

    # convert column ''customer_reviews' to string list format with separated comments, for those with Nan converted to 'no_review'
    df['customer_reviews'] = df['customer_reviews'].apply(lambda x: x.split('|') if type(x)==str else ['no_review'])
    df['customer_reviews'] = df['customer_reviews'].apply(lambda x: ','.join(x))

    return df

def clean_seller_col(df):
    # convert the column 'sellers' to a new column 'seller_info_list', in where data format 'seller1,price1,seller2,price2,....', 'seller1,price1,..'
    seller_info_list = []
    for row in df['sellers']:
        sub_list = []
        if type(row) == str:
            for i in row.replace('=>[{','').split(','):
                i = i.replace('=>{','')
                re_i = ''.join(re.findall('=>(.*)', i))
                try:
                    re_i = re_i.replace('}','').replace(']','').replace('"', '').replace("'","")
                except IndexError:
                    re_i = re_i
                sub_list.append(re_i)
        else:
            sub_list.append('no_seller_info')
        sub_list = ','.join(sub_list)
        seller_info_list.append(sub_list)

    df['seller_info_list'] = seller_info_list

    return df

def fill_na_price_and_integrate_cols(df):
    # fill value = 0 to Nan entries in columns 'lower_price' and 'upper_price' for calculating the avg_price
    df['lower_price'] = df['lower_price'].fillna(value=0)
    df['upper_price'] = df['upper_price'].fillna(value=0)


    # integrate the columns to new dataframe named 'clean_comb_df'
    clean_comb_df = df[['uniq_id',
        'product_name',
        'manufacturer',
        'number_available_in_stock',
        'number_of_reviews',
        'number_of_answered_questions',
        'average_review_rating',
        'amazon_category_and_sub_category',
        'product_information',
        'customer_questions_and_answers',
        'customer_reviews',
        'lower_price',
        'upper_price',
        'avg_price',
        'price_tag',
        'seller_info_list']]

    return clean_comb_df


def product_name_token(clean_comb_df):
    """
    input: a df with a column of product_names ; return: a df with a new col 'product_names_tokens'
    """
    product_names = clean_comb_df['product_name']
    product_names_tokens_list = []
    for n in product_names:
        name_tokens = word_tokenize(n)
        name_tokens = ','.join([t for t in name_tokens if ((t not in list(punctuation)) and (t not in list(stopwords.words('english'))))])
        product_names_tokens_list.append(name_tokens)

    clean_comb_df['product_name_tokens'] = product_names_tokens_list
    return clean_comb_df



def seller_list_token(clean_comb_df):
    """
    input: a column of products_sellers_list ; return: a list of seller_list_tokens
    """
    products_sellers_list = clean_comb_df['seller_info_list']
    sellers = []
    sellers_list = products_sellers_list.apply(lambda x: x.split(','))
    for i in sellers_list:
        sellers.append(','.join([i[j] for j in range(len(i)) if j%2==0]))

    clean_comb_df['seller_list_tokens'] = sellers
    return clean_comb_df



def product_info_token(clean_comb_df):
    """
    input: a column of products_info ; return a list of product_info_tokens
    """
    # setup a stopwords list here
    _stopwords = set(stopwords.words('english') + list(punctuation))
    products_info = clean_comb_df['product_information']
    products_info_list = []
    for product in products_info:
        try:
            products_info_list.append([token for token in word_tokenize(product) if ((token not in _stopwords) and (token.replace('.','').replace(',','').isnumeric()==False) and len(token) > 1)])
        except TypeError:
            products_info_list.append([])

    clean_comb_df['product_info_token'] = products_info_list
    clean_comb_df['product_info_token'] = clean_comb_df['product_info_token'].apply(lambda x: ','.join(x))
    return clean_comb_df


def combine_tokens(clean_comb_df):
    """
    combine all the tokens together in a new column named 'combined_tokens'
    product_name_tokens
    amazon_category_and_sub_category
    price_tag
    seller_list_tokens
    product_info_token
    """
    combined_tokens = []
    for i in range(clean_comb_df.shape[0]):
        empty = ''
        empty += clean_comb_df['product_name_tokens'][i]
        empty += clean_comb_df['amazon_category_and_sub_category'][i]
        empty += clean_comb_df['price_tag'][i]
        empty += clean_comb_df['seller_list_tokens'][i]
        empty += clean_comb_df['product_info_token'][i]

        combined_tokens.append(empty)
    clean_comb_df['combined_tokens'] = combined_tokens
    return clean_comb_df


# for exporting the ready-to-use DataFrame:
df_1 = price_tag_converter(df)
df_2 = clean_some_cols(df_1)
df_3 = clean_seller_col(df_2)
df_4 = fill_na_price_and_integrate_cols(df_3)
df_5 = product_name_token(df_4)
df_6 = seller_list_token(df_5)
df_7 = product_info_token(df_6)
final_df = combine_tokens(df_7)

final_df.to_csv('final_df.csv')