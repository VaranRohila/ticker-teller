from flask import Flask, jsonify
from pymongo import MongoClient
from bson.json_util import dumps
from bson import ObjectId
from src.mongo_scripts.mongo_read import MongoRead

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
        return jsonify(companies), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/article/<articleId>', methods=['GET'])
def get_article_data(articleId):
    try:
        articleData = mongoRead.get_news_articles_by_goid(articleId)
        articleData = serialize_doc(articleData) 
        
        return jsonify(articleData), 200
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
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/predictStockData/<companyId>/<date>')
def predictStockData(companyId, date):
    
    # Get stock data
    stock_data = mongoRead.get_stock_series_for_last_month(companyId, date)
    stock_data = serialize_doc(stock_data)

    # 

    return jsonify(stock_data), 200


if __name__ == '__main__':
    app.run(debug=True)
