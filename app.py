from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# preprocess
def preprocess(review):
    review = BeautifulSoup(review, "html.parser").get_text()  
    review = re.sub(r"[^a-zA-Z]", " ", review.lower())  
    tokens = word_tokenize(review) 
    filtered_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))  
        for word, tag in pos_tag(tokens)  
        if word not in stop_words 
    ]
    return filtered_tokens

word2vec = joblib.load('word2vec.pkl')

# fortosi tou neural
model = joblib.load('neural.pkl')  
def vectorize_review(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


# arxikopoiisi flask app
app = Flask(__name__)




# route homepage
@app.route('/')
def home(): 
    return render_template('index.html')

# route gia provlepsi
@app.route('/predict', methods=['POST'])
def predict():
    #pare tin kritiki apo to form
    review = request.form['review']
    
    
    
    preproc = preprocess(review)
    vectorized = vectorize_review(preproc,word2vec,200)
    print('\nShape of Tensor:', tf.shape(vectorized).numpy())
    reshaped_tensor = tf.reshape(vectorized, [-1, 200])
    print('\nShape of Tensor NEW:', tf.shape(reshaped_tensor).numpy())
    
    prediction = model.predict(reshaped_tensor)
    print(prediction)
    # apokodikopoiise to apotelesma
    result = 'Good Review' if prediction[0] > 0.8 else 'Bad Review'
    
    # epestrepse to apotelesma
    return render_template('index.html', prediction_text=f'This is a {result}')

if __name__ == '__main__':
    app.run(debug=True)

