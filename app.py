from itertools import product
from os import name
import re
from flask import Flask, redirect, url_for, render_template,request , Markup
import pickle
from flask import Flask
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from flask_sqlalchemy import SQLAlchemy
from wordcloud import WordCloud # for word cloud



app = Flask(__name__)
# Add Database
app.config['SQLALCHEMY_DATABSE_URI'] = 'sqlite://users.db'
@app.route("/")
def index():
    title = 'My Portfolio'
    return render_template('index.html',title=title)


model = pickle.load(open('cos_similarity.pkl','rb'))
df=pd.read_csv('data/final_df.csv')


@app.route("/prediction") 
def prediction():
    price_list = sorted(df['price_tag'].value_counts().index.tolist())
    print(price_list)
    return render_template('prediction.html',price_list=price_list)

## Recommendation System
## Basic Search function -------------------------------------------------------
## different execution considering the checkbox checked or not
def decision(checked,brand_name,price_range):
    checked = checked # 1 = yes, 0 = no
    if checked:
        include_index = df.loc[(df['manufacturer']!=brand_name)&(df['price_tag']==price_range)].index.values
    else:
        include_index = df.loc[(df['manufacturer']==brand_name)&(df['price_tag']==price_range)].index.values
    return include_index

def filter_product(checked,scores, brand_name,price_range):
    x = [i[0] for i in scores]
    y = [scores[x.index(j)] for j in decision(checked,brand_name,price_range) if j in x]
    print(y[:10])
    return y

def basic_search_recommend(name,checked,price_range): 
    recommend_list = []
    makeup_id = df[df['product_name']==name].index.values[0]
    brand_name = df[df['product_name']==name]['manufacturer'][0]


    scores = list(enumerate(model[makeup_id]))
    print(scores[:10])
    y = filter_product(checked,scores,brand_name,price_range)
    sorted_scores = sorted(y, key = lambda x: x[1], reverse=True)
    sorted_scores = sorted_scores[:10]
    print(sorted_scores)
    

    j = 0
    print('The 10 most recommened products: ')
    for item in sorted_scores:
        product_name = df.iloc[item[0],:]['product_name']
        recommend_list.append(product_name)
        j = j+1
        if j > 9:
            break
    return recommend_list



@app.route("/basic_search_result",methods=['POST'])
def basic_search_result():
    selling_product = request.form.get('selling_product')
    price_range = request.form.get('price_range')
    checked = request.form.get('checked')
    prediction = basic_search_recommend(selling_product,checked,price_range)
    manufacturer_list=[df[df['product_name']==manufacturer]['manufacturer'].values[0] for manufacturer in prediction]
    price_list=[df[df['product_name']==price]['price_tag'].values[0] for price in prediction]
    sentiment_list=[df[df['product_name']==product]['summary_sentiment'].values[0] for product in prediction]
    return render_template('basic_search_result.html',prediction=prediction,manufacturer_list=manufacturer_list,price_list=price_list,sentiment_list=sentiment_list)

## -----------------------------------------------------

## Advanced Search Function-----------------------------
def keywords(name): 
    try:
        keywords_product = df[df['product_name'].str.contains(name, case=False)]['product_name'].head(20)
    except:
        try: 
            keywords_product  = df[df['manufacture'].str.contains(name, case=False)]['product_name'].head(20)
        except:
            try:
                keywords_product  = df[df['categories'].str.contains(name, case=False)]['product_name'].head(20)
            except:
                return 'No suitable products'
    return keywords_product.values.tolist() 

@app.route("/advanced_search_result",methods=['POST'])
def advanced_search_result():
    name = request.form.get('keywords') 
    price_range = request.form.get('price_range')
    selections = keywords(name)
    return render_template('advanced_search_result.html',selections=selections,name=name,price_range=price_range)

def advance_search_recommend(name): 
    recommend_list = []
    makeup_id = df[df['product_name']==name].index.values[0]
    scores = list(enumerate(model[makeup_id]))
    print(scores[1:11])
    sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
    sorted_scores = sorted_scores[1:11]
    print(sorted_scores)
    
    j = 0
    print('The 10 most recommened products: ')
    for item in sorted_scores:
        product_name = df.iloc[item[0],:]['product_name']
        recommend_list.append(product_name)
        j = j+1
        if j > 9:
            break
    return recommend_list

@app.route("/advanced_search_result2",methods=['POST'])
def advanced_search_result2():
    selection = request.form.get('selection')
    prediction = advance_search_recommend(selection)
    manufacturer_list=[df[df['product_name']==manufacturer]['manufacturer'].values[0] for manufacturer in prediction]
    price_list=[df[df['product_name']==price]['price_tag'].values[0] for price in prediction]
    sentiment_list=[df[df['product_name']==product]['summary_sentiment'].values[0] for product in prediction]
    return render_template('advanced_search_result2.html',prediction=prediction,manufacturer_list=manufacturer_list,price_list=price_list,sentiment_list=sentiment_list)

## -----------------------------------------------------

@app.route("/product_info",methods=['POST'])
def product_info():
    for key in request.form.keys():
        title = key
        print(title)

        re_text = df[df["product_name"] == title].customer_reviews
        re_comment = " ".join(re_text.astype("str"))
        wordcloud = WordCloud(background_color='white',width=600, height=600).generate(re_comment)
        # enlarge size, alia (html)
        # generate wordclouds for each item in product_name_list and add to img_list
        img = Markup(wordcloud.to_svg())

    return render_template('product_info.html',title=title,img=img)

