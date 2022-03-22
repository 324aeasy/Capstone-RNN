# There is also a url_for for the python side
from flask import Flask, render_template,session,redirect,url_for
from flask_wtf import FlaskForm
from wtforms import (StringField,BooleanField,DateTimeField,
                RadioField,SelectField,
                TextAreaField,SubmitField)
from wtforms.validators import DataRequired
import pickle
from nltk.corpus import stopwords
import re
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# Naive Bayes model import
# Combining non-ascii, punctuation, and stopword removal
def text_process(text):
    """
    1. remove non-ascii characters (â, €, ™, œ)
    2. remove punctuation
    3. remove stop words
    4. return list of lowercase clean text words
    """
    ascii_only = re.sub(r'[^\x00-\x7f]',r'', text)

    nopunc = [char for char in ascii_only if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]

### FUNCTIONS FOR RNN MODEL

# Remove non-ascii characters
def no_ascii(text):
    return (re.sub(r'[^\x00-\x7f]',r'', text))

# Capital letters function
def num_capital_letters(text):
    result = 0
    for i in text:
        if i.isupper():
            result += 1
    return result

# General number of punctuation
def all_punc_count(text):
    result = 0
    for i in text:
        if i in string.punctuation:
            result += 1
    return result

# For specific puncs
def punc_occurences(text, punc_mark, word_length):
    result = 0
    for i in text:
        if i == punc_mark:
            result += 1
    return result / word_length

def stopword_count(text, word_length):
    result = 0
    for word in text.lower().split():
        if word in stopwords.words('english'):
            result += 1
    return result / word_length

# Open naive pipeline savefile
with open('naivepipeline.sav', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    pipeline = u.load()

# Open RNN savefile
rnn_news = pd.read_csv('static/fullrnnfeatures.csv')
X = rnn_news.drop(['r_or_f','title','text', 'subject','date','real_0_fake_1'],axis=1).values
y = rnn_news['real_0_fake_1'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = tf.keras.models.load_model('static/fullrnnweights.h5')

# Takes in raw text and creates desired numpy array
def new_art_preprocessor(text):
    res = []
    res.append(len(text)) # 0: char_length
    res.append(len(text.split())) # 1: word_length
    res.append(res[0] / res[1]) # 2: avg_char_word
    res.append(num_capital_letters(text) / res[1]) # 3: capital_per_word_length
    res.append(all_punc_count(text) / res[1]) # 4: all_punc_count_word
    for punc in string.punctuation:
        res.append(punc_occurences(text,punc,res[1])) # 5-36: punctuations
    res.append(stopword_count(text,res[1])) # 37: stopwords

    art_array = pd.DataFrame(res).values
    return scaler.transform(art_array.reshape(-1,38))


###### Website
app = Flask(__name__)

app.config['SECRET_KEY'] = 'mykey'

# This is a complex form, which is why we usually create a separate form file.
class InfoForm(FlaskForm):

    # Sometimes this needs to be a unicode string, put 'u' in front of it.
    model_choice = SelectField(u'Which model do you want to use:',
            choices=[('naive_model_choice','Naive Bayes'),('tf_model_choice','RNN')])
    input_article = TextAreaField()
    submit = SubmitField('Submit')

@app.route('/',methods=['GET','POST'])
def index():

    form = InfoForm()
    # Checks for validators
    if form.validate_on_submit():
        # Can use session to store and retrieve temporary session information
        session['model_choice'] = form.model_choice.data
        session['input_article'] = form.input_article.data

        # Naive Bayes model chosen:
        if session['model_choice'] == 'naive_model_choice':
            # Does not allow blank entries.
            if len(session['input_article'].replace(' ','')) == 0:
                session['result_prob_real'] = 'Null'
                session['result_prob_fake'] = 'Null'
                session['result'] = 'Please enter a non-empty input'
            else:
                session['result'] = str(pipeline.predict([session['input_article']])[0])[1:][1:-1]
                probability_list = list(pipeline.predict_proba([session['input_article']])[0])
                session['result_prob_fake'] = f"{round(probability_list[0]*100,2)}%"
                session['result_prob_real'] = f"{round(probability_list[1]*100,2)}%"
        elif session['model_choice'] == 'tf_model_choice':
            # Does not allow blank entries.
            if len(session['input_article'].replace(' ','')) == 0:
                session['result_prob_real'] = 'Null'
                session['result_prob_fake'] = 'Null'
                session['result'] = 'Please enter a non-empty input'
            else:
                score = model.predict_proba(new_art_preprocessor(session['input_article']))[0][0]
                session['result_prob_real'] = round((1-score)*100.0,2)
                session['result_prob_fake'] = round((score)*100.0,2)
                session['result'] = 'Real' if session['result_prob_real'] > session['result_prob_fake'] else 'Fake'
                session['result_prob_real'] = f"{session['result_prob_real']}%"
                session['result_prob_fake'] = f"{session['result_prob_fake']}%"
        # Use redirect and url_for to redirect without messing with htnml
        # Runs after valid submission
        return redirect(url_for('result'))

    # Overall return for original rendering
    return render_template('index.html',form=form)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/credits')
def credits():
    return render_template('credits.html')

if __name__ == '__main__':
    app.run(debug=True)
