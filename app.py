from flask import Flask,render_template,flash,redirect,url_for, session,request,logging,jsonify
# from flask_mysqldb import MySQL
from wtforms import Form,StringField,TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from flask import Response
import matplotlib.pyplot as plt
import pandas as pd
import pickle
app = Flask(__name__)

app.secret_key = 'many random bytes'

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# class review_predict(Form):
#     review = StringField('Review', [validators.Length(min=1,max=80)])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def analyze_sentiment(df):
    sentiments = []
    sid = SentimentIntensityAnalyzer()
    for i in range(df.shape[0]):
        line = df['Review'].iloc[i]
        sentiment = sid.polarity_scores(line)
        sentiments.append([sentiment['neg'], sentiment['pos'],
                           sentiment['neu'], sentiment['compound']])
    df[['neg', 'pos', 'neu', 'compound']] = pd.DataFrame(sentiments)
#     df['Negative'] = df['compound'] < -0.1
#     df['Positive'] = df['compound'] > 0.1
    return df

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/', methods = ["POST"])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        if review is None:
            print("Enter your review first.")
        else:
             data = [review]
             test = pd.DataFrame(data, columns = ['Review'])
             df1 = analyze_sentiment(test)
             review = vectorizer.transform(data)
             testt = pd.DataFrame(review.toarray())
             df_finaltest = pd.concat([testt, df1], axis =1)
             df_finaltest = df_finaltest.drop('Review', axis = 1)
             prediction = model.predict(df_finaltest)
    #output = prediction[0]
    return render_template("prediction.html",output ="Your review was predicted as " +prediction[0])


# @app.route('/api',methods=['POST'])
# def predict():
#     # Get the data from the POST request.
#     data = request.get_json(force=True)
#     data = [data['review']]
#     prediction = model.predict(vectorizer.transform(data))
#     output = prediction[0]
#     return jsonify(output)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about2')
def about2():
    return render_template('about2.html')
@app.route('/contact2')
def contact2():
    return render_template('contact2.html')
@app.route('/drugs')
def drugs():
    return render_template('drugs.html')
@app.route('/lda')
def lda():
    return render_template('lda.html')
@app.route('/bokehapp')
def bokehapp():
    return render_template('bokehapp.html')
@app.route('/review-2')
def review_2():
    return render_template('review-2.php')
@app.route('/review-3')
def review_3():
    return render_template('review-3.php')
@app.route('/review-4')
def review_4():
    return render_template('review-4.php')
@app.route('/review-5')
def review_5():
    return render_template('review-5.php')
@app.route('/review-6')
def review_6():
    return render_template('review-6.php')
@app.route('/review-7')
def review_7():
    return render_template('review-7.php')
@app.route('/review-8')
def review_8():
    return render_template('review-8.php')


    
@app.route('/summary')
def summary():
    return render_template('drug_list.html')

@app.route('/summary', methods=['POST'])
def my_form_post():
    text = request.form['text']
    file_name = text
    file_name=file_name+'.csv'
    session['filename']=file_name
    session['fname']=text
    return redirect(url_for('view_summary'))

@app.route('/view_summary')
def view_summary():
    return top_sentences()
def top_sentences():
    import pandas as pd
    file_name=session['filename']
    reviews = pd.read_csv(file_name)
    effective=[]
    for index, row in reviews.iterrows():
        str=row['Review']+'.'
        if (row['Category']=='Effective'):
            effective.append(str)
    outF = open("effective_doc.txt", "w+")
    for line in effective:
      # write line to output file
      outF.write(line)
      outF.write("\n")
    outF.close()
    with open("effective_doc.txt") as f:
        document = f.read()
        document = ' '.join(document.strip().split('\n'))
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(document)
        from collections import Counter
 
        def bag_of_words(sentence):
            return Counter(word.lower().strip('.,') for word in sentence.split(' '))
        from sklearn.feature_extraction.text import CountVectorizer
        c = CountVectorizer()
        bow_array = c.fit_transform([sentences[0]])
        bow_array.toarray()
        from sklearn.feature_extraction.text import CountVectorizer
        c = CountVectorizer()
        bow_matrix = c.fit_transform(sentences)
        #rows representing sentences and columns representing words
        bow_matrix
        #Tf-idf for similarity graph 
        from sklearn.feature_extraction.text import TfidfTransformer
        normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
        #similarity between each sentence
        similarity_graph = normalized_matrix * normalized_matrix.T
        similarity_graph.toarray()
        #PageRank on graph to calculate importance of each node
        import networkx as nx
        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores=[]
        scores = nx.pagerank(nx_graph)
        ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
        your_list_effective=[ranked[0][1],ranked[1][1],ranked[2][1],ranked[3][1],ranked[4][1]]
        score=[]
        score=sorted(scores.values(),reverse=True)
        scores_eff=[score[0],score[1],score[2],score[3],score[4]] 
        
        file_name=session['filename']
    reviews = pd.read_csv(file_name)
    adverse=[]
    for index, row in reviews.iterrows():
        str=row['Review']+'.'
        if (row['Category']=='Adverse'):
            adverse.append(str)
    outF = open("adverse_doc.txt", "w+")
    for line in adverse:
      # write line to output file
      outF.write(line)
      outF.write("\n")
    outF.close()
    with open("adverse_doc.txt") as f:
        document = f.read()
        document = ' '.join(document.strip().split('\n'))
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(document)
        from collections import Counter
 
        def bag_of_words(sentence):
            return Counter(word.lower().strip('.,') for word in sentence.split(' '))
        from sklearn.feature_extraction.text import CountVectorizer
        c = CountVectorizer()
        bow_array = c.fit_transform([sentences[0]])
        bow_array.toarray()
        from sklearn.feature_extraction.text import CountVectorizer
        c = CountVectorizer()
        bow_matrix = c.fit_transform(sentences)
        #rows representing sentences and columns representing words
        bow_matrix
        #Tf-idf for similarity graph 
        from sklearn.feature_extraction.text import TfidfTransformer
        normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
        #similarity between each sentence
        similarity_graph = normalized_matrix * normalized_matrix.T
        similarity_graph.toarray()
        #PageRank on graph to calculate importance of each node
        import networkx as nx
        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores2 = nx.pagerank(nx_graph)
        ranked2 = sorted(((scores2[i],s) for i,s in enumerate(sentences)),reverse=True)
        your_list_adverse=[ranked2[0][1],ranked2[1][1],ranked2[2][1],ranked2[3][1],ranked2[4][1]]
        score_adr=sorted(scores2.values(),reverse=True)
        scores_adr=[score_adr[0],score_adr[1],score_adr[2],score_adr[3],score_adr[4]]
        
        reviews = pd.read_csv(file_name)
    ineffective=[]
    for index, row in reviews.iterrows():
        str=row['Review']+'.'
        if (row['Category']=='Ineffective'):
            ineffective.append(str)
    outF = open("ineffective_doc.txt", "w+")
    for line in ineffective:
      # write line to output file
      outF.write(line)
      outF.write("\n")
    outF.close()
    with open("ineffective_doc.txt") as f:
        document = f.read()
        document = ' '.join(document.strip().split('\n'))
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(document)
        from collections import Counter
 
        def bag_of_words(sentence):
            return Counter(word.lower().strip('.,') for word in sentence.split(' '))
        from sklearn.feature_extraction.text import CountVectorizer
        c = CountVectorizer()
        bow_array = c.fit_transform([sentences[0]])
        bow_array.toarray()
        from sklearn.feature_extraction.text import CountVectorizer
        c = CountVectorizer()
        bow_matrix = c.fit_transform(sentences)
        #rows representing sentences and columns representing words
        bow_matrix
        #Tf-idf for similarity graph 
        from sklearn.feature_extraction.text import TfidfTransformer
        normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
        #similarity between each sentence
        similarity_graph = normalized_matrix * normalized_matrix.T
        similarity_graph.toarray()
        #PageRank on graph to calculate importance of each node
        import networkx as nx
        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores3 = nx.pagerank(nx_graph)
        ranked3 = sorted(((scores3[i],s) for i,s in enumerate(sentences)),reverse=True)
        lgt=len(ranked3)
        your_list_ineffective=[]
        if (lgt<5):
            for i in range(0,lgt):
                your_list_ineffective.append(ranked3[i][1])
        else:
             your_list_ineffective=[ranked3[0][1],ranked3[1][1],ranked3[2][1],ranked3[3][1],ranked3[4][1]]
        score_ineff=sorted(scores3.values(),reverse=True)
        scores_ineff=[]
        if (lgt<5):
            for i in range(0,lgt):
                scores_ineff.append(score_ineff[i])
        else:
            scores_ineff=[score_ineff[0],score_ineff[1],score_ineff[2],score_ineff[3],score_ineff[4]]
            lgt=5;
        
        return cons(your_list_effective,your_list_adverse,your_list_ineffective,scores_eff,scores_adr,scores_ineff,lgt)
    
def cons(your_list_effective,your_list_adverse,your_list_ineffective,scores_eff,scores_adr,scores_ineff,lgt):
    import pandas as pd
    dictionary=pd.read_csv('ADR_lexicon - ADR_lexicon.csv', header = None)
    with open('adverse_doc.txt') as f:
        document = f.read()
    data=document.splitlines()
    diction = dictionary[1].tolist()
    import re
    dic_list=[]
    for sentence in data:
        wordList = re.sub("[^\w]", " ",  sentence).split()
        for word in wordList:
            for dicti in diction:
                if word==dicti: 
                    dic_list.append(dicti)
    #print(dic_list)
    import collections
    counts = collections.Counter(dic_list)
    new_list = sorted(dic_list, key=lambda x: -counts[x])
    new_list1=[]
    for word in new_list:
        if word not in new_list1:
            new_list1.append(word)
    
    new_list2=[]
    for i in range(0,9):
        word=new_list1[i]
        new_list2.append(word)
        
    cnt=[]    
    for word in new_list2:
        cnt.append(dic_list.count(word))
    dname=session['fname']
    con_graph_url = con_graph(cons=new_list2,occurences=cnt);   
    return render_template('summary.html',g=con_graph_url, your_list1=your_list_effective,your_list2=your_list_adverse,your_list3=your_list_ineffective,score1=scores_eff,score3=scores_ineff,score2=scores_adr,cons=new_list2,occurences=cnt,drug=dname,length=lgt)


def con_graph(cons,occurences):
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    fig, ax = plt.subplots()
    img = io.BytesIO()
    colors = ['#4A235A', '#5B2C6F', '#6C3483', '#7D3C98','#8E44AD','#A569BD','#BB8FCE','#D2B4DE','#E8DAEF','#F4ECF7']
    ax.barh(cons, occurences,align='center',color=colors, ecolor='black')
    ax.set_yticks(cons)
    ax.set_yticklabels(cons)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('occurences')
    ax.set_title('Lookout for the side effects!')
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    con_graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(con_graph_url)


if __name__ == "__main__":
    app.run(debug=True)
