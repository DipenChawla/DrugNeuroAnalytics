import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

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

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    data = [data['review']]
    df1 = analyze_sentiment(data)
    review = vectorizer.transform([data])
    testt = pd.DataFrame(review.toarray())
    df_finaltest = pd.concat([testt, df1], axis =1)
    df_finaltest = df_finaltest.drop('Review', axis = 1)
    prediction = model.predict(vectorizer.transform(data))
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
