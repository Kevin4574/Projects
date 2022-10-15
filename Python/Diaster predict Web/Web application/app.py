from flask import Flask, render_template, url_for
import os
import pandas as pd
import plotly, json
import plotly.graph_objs as go
from sqlalchemy import create_engine
import joblib
import sqlite3
from plotly.graph_objs import Bar
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import re
import nltk
from flask import render_template, request, jsonify

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

path = r'wrangling_python/data'

app = Flask(__name__)

# define a function to tokenize and clean the feature
def tokenize(text):
    '''clean text and return cleaned tokens

    Args:
        text: original input text

    Returns:
        tokens/words: cleaned token after remove URL and lemmatization
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class toarray(BaseEstimator, TransformerMixin):
    '''convert data to array

    Args:
        X: data

    Returns:
        output: array format data

    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X.toarray())

class length(BaseEstimator, TransformerMixin):
    '''return the length of the data

    Args:
        X: data

    Returns:
        output: the length of the data

    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        return pd.DataFrame([len(txt) for txt in X])

# load data
engine = create_engine('sqlite:///' + path + '\ETL_Cleaned.db')
df = pd.read_sql_table('message',engine)

# load model
model = joblib.load(path + "/Best_Model")

# set up our applications
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # GRAPH 2 - category graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()