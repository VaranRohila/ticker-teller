import pandas as pd
from datetime import date, timedelta

from taipy.gui import Gui, notify
from src.mongo_scripts.mongo_read import MongoRead

import numpy as np

import joblib
import json

from tensorflow import keras

# -------- Mongo --------
mongo_uri = "mongodb+srv://stockprediction:stockprediction@stockprediction.v0m2cg8.mongodb.net/"

# Connect to the database and collection
mongoRead = MongoRead(mongo_uri, 'StockData')

companies = mongoRead.get_all_companies()
company_mapper = {k: v for k,v in zip([x['Company Name'] for x in companies], [x['ID'] for x in companies])}
company = companies = list(set([x['Company Name'] for x in companies]))

layout = {"margin": {"l": 220}}

page = """
<|toggle|theme|>

<|25 75|layout|gap=30px|
<|sidebar|
## Please **filter**{: .color-primary} here:

<|{company}|selector|lov={companies}|label=Select the Company|dropdown|on_change=on_filter|class_name=fullwidth|>

<|Predict Next|button|class_name=fullwidth|on_action=on_predict_next|>
|>
<main_page|
# 📈 TickerTeller **Dashboard**{: .color-primary}

<|1 1|layout|
<|
## **Simulated**{: .color-primary} Date Range: <|{CurrentDate.strftime('%m/%d/%y')}|> - <|{(CurrentDate + timedelta(days=30)).strftime('%m/%d/%y')}|>
|>
|>

<br/>

<|{stock_series}|chart|mode=lines|x=Date|y[1]=Close|y[3]=TextClose|mode[3]=text|text[3]=NewsSentiments|y[2]=Predictions|color[1]=red|color[2]=#00FF00|>

<|News Articles Table|expandable|not expanded|
<|{news_articles_df}|table|page_size=5|>
|>


|main_page>
|>

"""

def on_predict_next(state):
    if state.CurrentDate >= date(2024, 1, 31):
        notify(state, "Error", "Stock Data not available for the simulated date.")
        return


    curr_date = state.CurrentDate
    cid = company_mapper[state.company]

    stock_series = state.stock_series

    # Loads
    scaler = joblib.load(f'models/Scalers/{cid}.pkl')
    model = keras.models.load_model(f'models/LSTM/{cid}.h5')
    lr_model = joblib.load(f'models/Ensemble/lr_models/{cid}.pkl')
    lr_scaler = joblib.load(f'models/Ensemble/scalers/{cid}.pkl')

    for i in range(30):
        filter_df = stock_series[stock_series['Date'].dt.date == curr_date]
        if len(filter_df) > 0:
            output = predictStockData(cid, curr_date.strftime('%Y-%m-%d'), scaler, model, lr_model, lr_scaler)
            stock_series.loc[stock_series['Date'].dt.date == curr_date, 'Predictions'] = output
        
        curr_date = curr_date + timedelta(days=1)

    state.stock_series = stock_series
    state.CurrentDate = curr_date

def predictStockData(companyId, date, scaler, model, lr_model, lr_scaler):
    # Get stock data
    stock_data = mongoRead.get_stock_series_for_last_month(companyId, date)
    companyId = int(companyId)

    # Create Data for LSTM
    data = [float(x['Close']) for x in stock_data]
    data.reverse()
    data = scaler.transform(np.array(data).reshape(-1, 1))
    data = np.array([data])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

    # Load LSTM
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

        lr_pred = lr_model.predict([[llm_sentiment, llm_movement, lstm_pred]])
        output = lr_scaler.inverse_transform(lr_pred)[0][0]
    return output

def filter(company_name):
    cid = company_mapper[company_name]
    # Get News Articles
    api_response = mongoRead.get_news_articles(int(cid))
    news_articles_df = pd.DataFrame.from_dict(api_response)
    news_articles_df['Date'] = pd.to_datetime(news_articles_df['Date'])

    # Get Stock Series
    api_response = mongoRead.get_stock_series(int(cid))
    stock_series = pd.DataFrame.from_dict(api_response)
    stock_series['Date'] = pd.to_datetime(stock_series['Date'])

    stock_series['Predictions'] = None
    stock_series['NewsSentiments'] = None
    stock_series['TextClose'] = stock_series['Close'] + 40

    for _, row in news_articles_df.sort_values(by='impact_score', ascending=False).iloc[:10, :].iterrows():
        stock_series.loc[
                (stock_series['CID'] == row['CID']) &
                (stock_series['Date'].dt.date == row['Date'].date())
            , 'NewsSentiments'] = row['sentiment']
    
    news_articles_df = news_articles_df.drop(columns='CID')
    return news_articles_df, stock_series

def on_filter(state):
    state.news_articles_df, state.stock_series = filter(
        state.company
    )

    state.CurrentDate = date(2023, 8, 1)
    if len(state.news_articles_df) == 0:
        notify(state, "Error", "No results found. Check the filters.")
        return
    
if __name__ == "__main__":
    company = companies[0]
    news_articles_df, stock_series = filter(
        company
    )
    CurrentDate = date(2023, 8, 1)
    Gui(page).run(margin="0em", title="TickerTeller Dashboard")