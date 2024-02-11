from flask import Flask, jsonify, render_template
from pymongo import MongoClient
from bson.json_util import dumps
from bson import ObjectId
from src.mongo_scripts.mongo_read import MongoRead

import numpy as np
import gc

import joblib

from tensorflow import keras
import json


app = Flask(__name__)


mongo_uri = "mongodb+srv://stockprediction:stockprediction@stockprediction.v0m2cg8.mongodb.net/"
client = MongoClient(mongo_uri)

# Connect to the database and collection
db = client.StockData
mongoRead = MongoRead(mongo_uri, 'StockData')

# Utility function to convert ObjectId to string
def serialize_doc(doc):
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if not isinstance(doc, dict):
        return doc
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            doc[key] = str(value)
        elif isinstance(value, dict):
            doc[key] = serialize_doc(value)
        elif isinstance(value, list):
            doc[key] = [serialize_doc(item) for item in value]
    return doc



##### API Routes ######
@app.route('/', methods=['GET'])
def get_companies():
    try:
        companies_collection = db.Companies
        companies = list(companies_collection.find({}, {'_id': 0}))  # Exclude the '_id' field
        return render_template('index.html', companies=companies)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/article/<articleId>', methods=['GET'])
def get_article_data(articleId):
    try:
        articleData = mongoRead.get_news_articles_by_goid(articleId)
        return render_template('article.html', article = articleData[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/newsArticleData/<companyId>')
def getNewsArticlesForCompany(companyId):
    try:
        articles_by_date = mongoRead.get_news_articles_by_date(companyId)
        articles_by_impact_factor = mongoRead.get_news_articles_by_impact_score(companyId)

        articles_by_date = serialize_doc(articles_by_date)
        articles_by_impact_factor = serialize_doc(articles_by_impact_factor)

        result = {
            "ArticlesByDate": articles_by_date,
            "ArticlesByImpactFactor": articles_by_impact_factor
        }
        return render_template('result.html', result = result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predictStockData/<companyId>/<date>')
def predictStockData(companyId, date):
    # Get stock data
    stock_data = mongoRead.get_stock_series_for_last_month(companyId, date)
    stock_data = serialize_doc(stock_data)
    companyId = int(companyId)

    # Load Scaler
    scaler = joblib.load(f'models/Scalers/{companyId}.pkl')

    # Create Data for LSTM
    data = [float(x['Close']) for x in stock_data]
    data.reverse()
    data = scaler.transform(np.array(data).reshape(-1, 1))
    data = np.array([data])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    
    # Load LSTM
    model = keras.models.load_model(f'models/LSTM/{companyId}.h5')
    lstm_pred = model.predict(data)
    output = lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]

    # Get News Article for today
    news_article = mongoRead.get_news_articles_filter_date(companyId, date)
    if len(news_article) > 0:
        llm_sentiment = float(news_article[0]['sentiment'])
        llm_movement = news_article[0]['stock_movement']

        if llm_movement == 'Up': llm_movement = 1
        elif llm_movement == 'Down': llm_movement = -1
        else: llm_movement = 0

        lr_model = joblib.load(f'models/Ensemble/lr_models/{companyId}.pkl')
        lr_pred = lr_model.predict([[llm_sentiment, llm_movement, lstm_pred]])

        lr_scaler = joblib.load(f'models/Ensemble/scalers/{companyId}.pkl')
        output = lr_scaler.inverse_transform(lr_pred)[0][0]

    return jsonify({'model_prediction': str(output)}), 200

if __name__ == '__main__':
    app.run(debug=True)
