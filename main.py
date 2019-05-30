
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
def SearchResults():
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

    d = [0,0,0]
    d = [df.loc[[a]],df.loc[[b]],df.loc[[c]]]
    result = [d[0].iloc[0][2],d[1].iloc[0][2],d[2].iloc[0][2]]
	
    #Copy pasting the html doc so that it is just one file for viewing on github
    return ' <link rel="stylesheet" type="text/css" href="static/home.css"><div class="w3-top"><div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;"><a href="#home" class="w3-bar-item w3-button">Meme Me</a><div class="w3-right w3-hide-small"><a href="#about" class="w3-bar-item w3-button">About</a><a href="#contact" class="w3-bar-item w3-button">Contact</a></div></div></div><p class="w3-large"><font face="verdana" ></br></br></br></br>Your results are:<br/> <a href=%s> Result 1</a> " with a score of %f" <a href=http://thekipz.pythonanywhere.com/Recommend/%d>Recommended</a><br/><a href=%s> Result 2</a>" with a score of %f"<a href=http://thekipz.pythonanywhere.com/Recommend/%d>Recommended</a><br/> <a href=%s> Result 3</a>" with a score of %f"<a href=http://thekipz.pythonanywhere.com/Recommend/%d>Recommended</a><br/> <a href="/">Back Home</a></font><span class="w3-tag w3-light-grey"></span></p>' % ( result[0], score[0],a,result[1],score[1],b, result[2],score[2],c)

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
    #Copy pasting the html doc so that it is just one file for viewing on github
    return ' <link rel="stylesheet" type="text/css" href="static/home.css"><div class="w3-top"><div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;"><a href="#home" class="w3-bar-item w3-button">Meme Me</a><div class="w3-right w3-hide-small"><a href="#about" class="w3-bar-item w3-button">About</a><a href="#contact" class="w3-bar-item w3-button">Contact</a></div></div></div><p class="w3-large"><font face="verdana" ></br></br></br></br>Your query was classified as %s </a><br/> <a href="/">Back Home</a></font><span class="w3-tag w3-light-grey"></span></p>' % (clf.predict(X[z]))


@app.route('/Recommend/<int:post_id>')
def Recommend(post_id):


    df = pandas.read_csv('XData.csv', encoding = 'latin1')
    cat = df.caption



    # initialize vectorizer and convert text to vectors
    vectorizer = TfidfVectorizer(use_idf = True)
    X = vectorizer.fit_transform(cat)

    # compute cosine similarity between vectors
    sim = cosine_similarity(X)
    sim[post_id,post_id] = 0
    score = [0,0,0]
    a = numpy.argmax(sim[post_id])
    score[0] = sim[post_id,a]
    sim[post_id,a]=0
    b = numpy.argmax(sim[post_id])
    score[1] = sim[post_id,b]
    sim[post_id,b]=0
    c = numpy.argmax(sim[post_id])
    score[2] = sim[post_id,c]

    d = [0,0,0]
    d = [df.loc[[a]],df.loc[[b]],df.loc[[c]]]

    result = [d[0].iloc[0][2],d[1].iloc[0][2],d[2].iloc[0][2]]
	
    #Copy pasting the html doc so that it is just one file for viewing on github
    return ' <link rel="stylesheet" type="text/css" href="static/home.css"><div class="w3-top"><div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;"><a href="#home" class="w3-bar-item w3-button">Meme Me</a><div class="w3-right w3-hide-small"><a href="#about" class="w3-bar-item w3-button">About</a><a href="#contact" class="w3-bar-item w3-button">Contact</a></div></div></div><p class="w3-large"><font face="verdana" ></br></br></br></br>Your results are:<br/> <a href=%s> Result 1</a> " with a score of %f"<br/> <a href=%s> Result 2</a>" with a score of %f"<br/> <a href=%s> Result 3</a>" with a score of %f"<br/> <a href="/">Back Home</a></font><span class="w3-tag w3-light-grey"></span></p>' % ( result[0], score[0],result[1],score[1], result[2],score[2])
