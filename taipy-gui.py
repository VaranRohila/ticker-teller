import pandas as pd
from datetime import date, timedelta

from taipy.gui import Gui, notify
from src.mongo_scripts.mongo_read import MongoRead

import numpy as np

import joblib
import json
from sklearn.metrics import mean_squared_error

import ast

from tensorflow import keras

# -------- Mongo --------
mongo_uri = "mongodb+srv://stockprediction:stockprediction@stockprediction.v0m2cg8.mongodb.net/"

# Connect to the database and collection
mongoRead = MongoRead(mongo_uri, 'StockData')

companies = mongoRead.get_all_companies()
company_mapper = {k: v for k,v in zip([x['Company Name'] for x in companies], [x['ID'] for x in companies])}
company = companies = list(set([x['Company Name'] for x in companies]))

layout = {
    "margin": {"l": 220, "r": 220}
}

chart_layout = {
    "xaxis": {
        "title": "Date"
    },
    "yaxis": {
        "title": "Stock Price"
    }
}
page = """
<|toggle|theme|>
<|25 75|layout|gap=30px|
<|sidebar|
#### View **NewsBuddy**{: .color-primary} Insights:
<|{current_goid}|selector|lov={news_goids}|label=Select Article By ID|dropdown|on_change=on_goid_filter|class_name=fullwidth|>
<small> Powered by **GPT-3.5-Turbo**{: .color-primary} </small>
<hr/>
<b> Sentiment Evidence: </b> 
<br/>
<|{evidence}|>
<br/>
<br/>
<b> Predicted Stock Movement by Article: </b> **<|{predictive_stock_movement}|>**{: .color-primary}
<br/>
<br/>
<b> Explanation: </b> <|{explaination}|>
|>
<main_page|
# 📈 TickerTeller **Dashboard**{: .color-primary}

<|1 1|layout|gap=60px|
<|
<|{company}|selector|lov={companies}|label=Select the Company|dropdown|on_change=on_filter|class_name=fullwidth|>
<|Predict Next|button|class_name=fullwidth|on_action=on_predict_next|>
|>
<|
**Simulated**{: .color-primary} Date Range: <|{CurrentDate.strftime('%m/%d/%y')}|> - <|{(CurrentDate + timedelta(days=30)).strftime('%m/%d/%y')}|id="header"|>
<br/>
<br/>
<br/>
**Mean**{: .color-primary} Squared Error: <|$ {round(mean_squared_error(actual_values, predictions) if len(predictions) > 0 else 0, 2)}|>
|>
|>
<br/>

<|{stock_series}|chart|mode=lines|x=Date|y[1]=Close|y[3]=TextClose|mode[3]=text|text[3]=NewsSentiments|y[2]=Predictions|color[1]=red|color[2]=#00FF00|layout={chart_layout}|>

<|News Articles Table|expandable|expanded|
<|{news_articles_df[['GOID', 'Title', 'Date', 'sentiment', 'impact_score']]}|table|page_size=5|>
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

    acc, preds = state.actual_values, state.predictions
    for i in range(30):
        filter_df = stock_series[stock_series['Date'].dt.date == curr_date]
        if len(filter_df) > 0:
            output = predictStockData(cid, curr_date.strftime('%Y-%m-%d'), scaler, model, lr_model, lr_scaler)
            stock_series.loc[stock_series['Date'].dt.date == curr_date, 'Predictions'] = output

            acc.extend([stock_series.loc[stock_series['Date'].dt.date == curr_date]['Close'].values[0]])
            preds.extend([output])
        
        curr_date = curr_date + timedelta(days=1)

    state.stock_series = stock_series
    state.CurrentDate = curr_date
    state.predictions = preds
    state.actual_values = acc

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
    news_articles_df = news_articles_df.sort_values(by='impact_score', ascending=False)
    news_articles_df['GOID'] = news_articles_df['GOID'].astype(str)
    news_articles_df['GOID'] = news_articles_df['GOID'].apply(lambda x: x[:-2])

    # Get Stock Series
    api_response = mongoRead.get_stock_series(int(cid))
    stock_series = pd.DataFrame.from_dict(api_response)
    stock_series['Date'] = pd.to_datetime(stock_series['Date'])

    stock_series['Predictions'] = None
    stock_series['NewsSentiments'] = None
    stock_series['TextClose'] = stock_series['Close'] + 30

    for _, row in news_articles_df.iloc[:10, :].iterrows():
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

    state.news_goids = state.news_articles_df.iloc[:5, :]['GOID'].values.tolist()
    state.current_goid = state.news_goids[0]
    state.evidence, state.predictive_stock_movement, state.explaination = getInsights(
        state.news_articles_df,
        state.current_goid
    )
    
    state.CurrentDate = date(2023, 8, 1)
    state.predictions = []
    state.actual_values = []
    if len(state.news_articles_df) == 0:
        notify(state, "Error", "No results found. Check the filters.")
        return


def getInsights(news_articles_df, goid):
    filter_df = news_articles_df[news_articles_df['GOID'] == goid]
    evidence = filter_df['evidence'].values[0]
    predictive_stock_movement = filter_df['stock_movement'].values[0]
    explaination = filter_df['explaination'].values[0]

    evidence = ast.literal_eval(evidence)
    processed_evidence = ''
    for i, sent in enumerate(evidence):
        processed_evidence += f'{i + 1}. ' + sent + '\n'

    evidence = processed_evidence
    return evidence, predictive_stock_movement, explaination
    
def on_goid_filter(state):
    state.evidence, state.predictive_stock_movement, state.explaination = getInsights(
        state.news_articles_df,
        state.current_goid
    )

if __name__ == "__main__":
    company = 'JPMorgan Chase & Co'
    predictions = []
    actual_values = []
    news_articles_df, stock_series = filter(
        company
    )
    news_goids = news_articles_df.iloc[:5, :]['GOID'].values.tolist()
    current_goid = news_goids[0]
    evidence, predictive_stock_movement, explaination = getInsights(news_articles_df, current_goid)
    CurrentDate = date(2023, 8, 1)
    Gui(page).run(margin="0em", title="TickerTeller Dashboard")