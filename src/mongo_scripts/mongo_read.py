import pymongo
from datetime import datetime, timedelta
from dateutil import parser

class MongoRead:
    def __init__(self, conn_string, db) -> None:
        client = pymongo.MongoClient(conn_string)
        self.db = client[db]
    
    def get_all_sectors(self):
        curr = self.db['Companies'].find({}, {'Sector': 1})
        return list(set([x['Sector'] for x in curr]))

    def get_all_companies(self):
        curr = self.db['Companies'].find()
        return list(curr)
    
    def get_companies(self, sector):
        query = {'Sector': sector}
        curr = self.db['Companies'].find(query)
        return list(curr)
    
    def get_stock_series(self, id):
        query = {'CID': id}
        curr = self.db['Companies'].find(query)
        return list(curr)
    
    def get_stock_series_for_last_month(self, id, currDate):
        currDate = parser.parse(currDate)
        prevDate = currDate - timedelta(days=30)
        query = {
            "CID": int(id),
            "Date": {
                "$gte": prevDate,
                "$lte": currDate 
            }
        }
        curr = self.db['StockSeries'].find(query).sort('Date', -1)
        return list(curr)

    def get_news_articles(self, id):
        query = {'CID': id}
        curr = self.db['NewsArticles'].find(query).sort("Date", -1)
        return list(curr)
    
    def get_news_articles_by_goid(self, id):
        query = {'GOID': int(id)}
        curr = self.db['NewsArticles'].find(query)
        return list(curr)
    
    def get_news_articles_by_date(self, id):
        query = {'CID': int(id)}
        curr = self.db['NewsArticles'].find(query).sort("Date", -1).limit(20)
        return list(curr)
    
    def get_news_articles_by_impact_score(self, id):
        query = {'CID': int(id)}
        curr = self.db['NewsArticles'].find(query).sort("impact_score", -1).limit(20)
        return list(curr)