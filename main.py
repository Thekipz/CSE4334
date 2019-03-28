
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas, numpy

from flask import Flask, request, render_template

app = Flask(__name__)

def Search(term):
    # load the dataset
    df = pandas.read_csv('XData.csv', encoding = 'latin1')
    cat = df.caption

	# convert query to series and add it to the corpus
    query = pandas.Series([term])
    cat = cat.append(query,ignore_index = True)

	# initialize vectorizer and convert text to vectors
    vectorizer = TfidfVectorizer(use_idf = True)
    X = vectorizer.fit_transform(cat)

	# compute cosine similarity between vectors
    sim = cosine_similarity(X)
    
    #Start classifier
    data = vectorizer.fit_transform(df.caption)
    z = X.shape[0]
    z -= 1
    trainX,testX,trainY,testY = train_test_split(X[0:z],df.polclass)
    # Model Generation Using Multinomial Naive Bayes
    clf= MultinomialNB().fit(X[0:z],df.polclass)
    predicted = clf.predict(testX)
    #End classifier

	# get the 3 highest similarity documents and return results
    size = sim.shape[0]
    size -= 1
    sim[size,size]=0
    a = numpy.argmax(sim[size])
    sim[size,a]=0
    b = numpy.argmax(sim[size])
    sim[size,b]=0
    c = numpy.argmax(sim[size])
    d = df.loc[[a]]
    result1 = d.iloc[0][2]
    d = df.loc[[b]]
    result2 = d.iloc[0][2]
    d = df.loc[[c]]
    result3 = d.iloc[0][2]
    return clf.predict(X[z]), result1, result2, result3


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/results', methods=['POST'])
def results():
    search = request.form['first_name']
    cat = Search(search)
	#Probably should add this html to a seperate document
    return ' <link rel="stylesheet" type="text/css" href="static/home.css"><div class="w3-top"><div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;"><a href="#home" class="w3-bar-item w3-button">Meme Me</a><div class="w3-right w3-hide-small"><a href="#about" class="w3-bar-item w3-button">About</a><a href="#contact" class="w3-bar-item w3-button">Contact</a></div></div></div><p class="w3-large"><font face="verdana" ></br></br></br></br>Your query was classified as %s and your results are:<br/> <a href=%s> Result 1</a> <br/> <a href=%s> Result 2</a><br/> <a href=%s> Result 3</a><br/> <a href="/">Back Home</a></font><span class="w3-tag w3-light-grey"></span></p>' % (cat)


