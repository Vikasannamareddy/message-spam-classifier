import pickle
from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import sklearn

ps=PorterStemmer()
ps.stem('running')
app = Flask(__name__)



def transform_msg(msg):
    msg = nltk.word_tokenize(msg)
    msg = " ".join(msg)
    msg = nltk.sent_tokenize(msg)
    msg = " ".join(msg)
    msg = msg.lower()#lower case
    msg = nltk.word_tokenize(msg)#tokenize
    
    y=[]
    for i in msg:               #removing special characters
        if i.isalnum():
            y.append(i)
    msg=y[:]
    y.clear()
    for i in msg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    msg=y[:]
    y.clear()
    for i in msg:
        y.append(ps.stem(i))
    return " ".join(y)
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model = pickle.load(open("./model.pkl","rb"))
        vectorizer = pickle.load(open("./vectorizer.pkl","rb"))
        message = request.form['message']
        preprocessed_message = transform_msg(message)
        vectorized_message = vectorizer.transform([preprocessed_message])
        prediction = model.predict(vectorized_message)[0]
        if prediction == 1:
            result = 'SPAM'
        else:
            result = 'NOT SPAM'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

