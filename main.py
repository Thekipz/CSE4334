
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import pandas, numpy

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/SearchResults', methods=['POST'])
def results():
    search = request.form['Search']
    # load the dataset
    df = pandas.read_csv('XData.csv', encoding = 'latin1')
    cat = df.caption

	# convert query to series and add it to the corpus
    query = pandas.Series([search])
    cat = cat.append(query,ignore_index = True)

	# initialize vectorizer and convert text to vectors
    vectorizer = TfidfVectorizer(use_idf = True)
    X = vectorizer.fit_transform(cat)

	# compute cosine similarity between vectors
    sim = cosine_similarity(X)

    #Start classifier
    z = X.shape[0]
    z -= 1
    # Model Generation Using Multinomial Naive Bayes
    clf= MultinomialNB().fit(X[0:z],df.polclass)
    #End classifier

	# get the 3 highest similarity documents and return results
    size = sim.shape[0]
    size -= 1
    sim[size,size]=0
    score = [0,0,0]
    a = numpy.argmax(sim[size])
    score[0] = sim[size,a]
    sim[size,a]=0
    b = numpy.argmax(sim[size])
    score[1] = sim[size,b]
    sim[size,b]=0
    c = numpy.argmax(sim[size])
    score[2] = sim[size,c]
    #score = [sim[size,a],sim[size,b],sim[size,c]]
    d = [0,0,0]
    d = [df.loc[[a]],df.loc[[b]],df.loc[[c]]]
    result = [d[0].iloc[0][2],d[1].iloc[0][2],d[2].iloc[0][2]]
	#Probably should add this html to a seperate document
    return ' <link rel="stylesheet" type="text/css" href="static/home.css"><div class="w3-top"><div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;"><a href="#home" class="w3-bar-item w3-button">Meme Me</a><div class="w3-right w3-hide-small"><a href="#about" class="w3-bar-item w3-button">About</a><a href="#contact" class="w3-bar-item w3-button">Contact</a></div></div></div><p class="w3-large"><font face="verdana" ></br></br></br></br>Your query was classified as %s and your results are:<br/> <a href=%s> Result 1</a> " with a score of %f"<br/> <a href=%s> Result 2</a>" with a score of %f"<br/> <a href=%s> Result 3</a>" with a score of %f"<br/> <a href="/">Back Home</a></font><span class="w3-tag w3-light-grey"></span></p>' % (clf.predict(X[z]), result[0], score[0],result[1],score[1], result[2],score[2])

@app.route('/ClassifyResults', methods=['POST'])
def Classify():
    search = request.form['Classify']
    # load the dataset
    df = pandas.read_csv('XData.csv', encoding = 'latin1')
    cat = df.caption

	# convert query to series and add it to the corpus
    query = pandas.Series([search])
    cat = cat.append(query,ignore_index = True)

	# initialize vectorizer and convert text to vectors
    vectorizer = TfidfVectorizer(use_idf = True)
    X = vectorizer.fit_transform(cat)
    #Start classifier
    z = X.shape[0]
    z -= 1
    # Model Generation Using Multinomial Naive Bayes
    clf= MultinomialNB().fit(X[0:z],df.polclass)
    #End classifier
    return ' <link rel="stylesheet" type="text/css" href="static/home.css"><div class="w3-top"><div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;"><a href="#home" class="w3-bar-item w3-button">Meme Me</a><div class="w3-right w3-hide-small"><a href="#about" class="w3-bar-item w3-button">About</a><a href="#contact" class="w3-bar-item w3-button">Contact</a></div></div></div><p class="w3-large"><font face="verdana" ></br></br></br></br>Your query was classified as %s </a><br/> <a href="/">Back Home</a></font><span class="w3-tag w3-light-grey"></span></p>' % (clf.predict(X[z]))
