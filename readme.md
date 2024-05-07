## TickerTeller | Hacklytics 2024
Project Link: https://devfolio.co/projects/tickerteller-2d2a

### **Project Overview: Solving the Investment Puzzle**

Our platform has been developed to tackle the challenge faced by prospective investors seeking accurate predictions of stock prices for their companies of interest. It integrates an ensemble machine learning model that combines LSTM-based time series forecasting and sentiment analysis from Wall Street Journal articles, offering insights into anticipated stock movements and the underlying reasons from the news. This holistic approach enables investors to access well-rounded predictions of future stock prices, enhanced by several additional functionalities designed to support informed investment decisions.

#### **Key Features:**
**Ensemble Machine Learning Forecasting Model:** At the heart of our platform is an ensemble model that merges the predictive capabilities of LSTM networks for Time Series Forecasting with sophisticated sentiment analysis. This model is specifically tailored to analyze time series data, gauge market sentiment from Wall Street Journal articles, and provide accurate future predictions.<br>
**Comprehensive Sentiment Analysis:** Understanding the impact of news on stock prices is crucial. Our platform does not just predict stock movements; it dives deep into the sentiment conveyed in financial news articles, providing a layer of insight that is often missed in traditional analysis.<br>
**Evidence-Based Predictions:** We go a step further by correlating our predictions with evidence extracted from the news articles. This approach not only enhances the reliability of our predictions but also offers investors a transparent view of the rationale behind predicted stock movements.<br>
**User-Friendly Dashboard:** All of these features are seamlessly integrated into an intuitive dashboard, making complex data easily accessible and understandable. Investors can quickly get a snapshot of future stock prices along with actionable insights, empowering them to make well-informed decisions.<br>

## Project Structure
```bash
│   app.py                  # Deprecated Flask API
│   favicon.ico
│   readme.md
│   requirements.txt        # Python Environment Specifications
│   taipy-gui.py            # Main UI of the application
├───data                    # For Data Needs, in .gitignore
│   ├───final
│   ├───intermediate
│   ├───meta
│   └───raw
├───models                  # Exported Models
│   ├───Ensemble            # Ensemble models with scaler
│   │   ├───lr_models
│   │   └───scalers
│   ├───LSTM                # LSTM models
│   └───Scalers             # LSTM scalers
├───notebooks               # Python Notebooks for building and testing
│       1. Data Preprocessing.ipynb
│       2. LLM.ipynb
│       2a. LLM Output Cleaning.ipynb
│       3. LLM - Test Code.ipynb
│       4. LSTM - Forecasting.ipynb
│       5. PyMongo.ipynb
│       6. ImpactScore.ipynb
│       7. Ensemble.ipynb
│       8. Predictions Analysis.ipynb
│       Z. ConvertCsvToJson.ipynb
├───outputs                 # Outputs for the models
│   ├───lstm_predictions    # LSTM Predictions for evaluation
│   ├───metrics             # Model Metrics
│   │       ensemble.csv
│   │       lstm.csv
│   │       result.csv
│   │
│   └───plots               # Graph Plots
├───src                     # Scripts for Deployment
│   ├───data
│   ├───features
│   ├───models
│   ├───mongo_scripts       # MongoDB Scripts
│   ├───utils
│   └───visualizations
```

## Run Locally
>Request MongoDB Atlas access from contributers.
```shell
pip install -r requirements.txt
```
Run Taipy UI
```shell
python taipy-gui.py
```

## Thank You Note
Thanks Hacklytics 2024 for this amazing oppurtunity.
