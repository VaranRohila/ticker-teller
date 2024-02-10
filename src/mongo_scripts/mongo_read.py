import pymongo

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

    def get_news_articles(self, id):
        query = {'CID': id}
        curr = self.db['NewsArticles'].find(query).sort("Date", -1)
        return list(curr)