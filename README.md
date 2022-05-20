# Search Traffic Financial Analysis and Forecast
## Overview & Features
At MercadoLibre, our mission is to operate online marketplaces dedicated to e-commerce and online auctions. To contribute to our mission, I have analyzed our companies financial and user data to enable growth and provide our leaders with forecasted data. 


This analysis utilizes the following steps:
1. Finding unusual patterns in hourly Google search traffic.
2. Mining the search traffic data for seasonality.
3. Relating the search traffic to stock price patterns.
4. Creating a time sereis model with Prophet.
5. Forecasting revenue using time series models.

![logo](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/unnamed.png)

---

## Technologies

This notebook utilizes **Python (v 3.9.7)** and the following libraries:

1. pandas
2. hvplot.pandas
3. dt from datetime
4. holoviews
5. Prophet from fbprophet
6. %matplotlib inline
7. pystan

This notebook is used in conjunction with Google Colab Notebooks. ##__Please click [here](https://colab.research.google.com/drive/12p9oihYAqmQ5YMZV7JvohKAwm1cveDHX?usp=sharing) to view in Google Colab.

---

## Installation Guide
Pandas and Pathlib should be part of the base applications that were installed with the Python version above; if not, you will have to install them through the pip package manager of Python.

To install pystan, run the following:

   
    pip install pystan
   

To install prophet, run the following:

    
    pip install fbprophet
    
To install holoviews, run the following:

    
    pip install holoviews
    
    
After installing, run the following to ensure that sklearn and hvplot are installed:

    
    conda list pystan
    
    conda list fbprophet
    
    conda list holoviews
    
    
If any errors occur, please contact IT for further assistance.

---

## User Guide
To use the notebook in Google Colab:

### Load the Data
1. Open "forecasting_net_prophet.ipynb"
2. Look for the following code:
    ```python
    from google.colab import files
    uploaded = files.upload()

    df_mercado_trends = pd.read_csv(
        "google_hourly_search_trends.csv",
        index_col="Date",
        parse_dates=True,
        infer_datetime_format=True
    )
    ```
Altering how we read a csv, we will use the ```files``` package within the ```google.colab``` library to create the ```uploaded = files.upload()``` function. This enables the user to select which .CSV file they would like to upload. The syntax below sets the "Date" column as the index and follows further DataFrame parameters.

### Focus the Data
Before analyzing the data, we need to focus our data. We want to look at May 2020, a time period when MercadoLibre released its financial results. How did this event affect our search traffic?
To do this, we did the following:
    ```python
    traffic_may_2020 = df_mercado_trends.loc["5/1/20":"5/31/20"].sum()
    
    median_monthly_traffic = df_mercado_trends["Search Trends"].groupby(by=[df_mercado_trends.index.year, df_mercado_trends.index.month]).sum().median()
    
    print(traffic_may_2020 / median_monthly_traffic)
    ```
    
Using these values, we observed that search traffic increased by 8% during the month that MercadoLibre released its financial results.

### Mine the Search Traffic Data for Seasonality
As a marketing company, we need to utilize every bit (or byte) of data at our disposal to drain as much ROI from our marketing budget as we can. To this end, I have created plots to show a level of seasonality in search traffic data.

Hourly Search Data by Day of Week                         |  Heatmap of Search Trends
:----------------------------------------:|:----------------------------------------:
![line](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/hourbyweek.PNG)  | ![heatmap](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/heatmap.PNG)

Why is this data relevant? Using the data, we observe that hours 23-02 on every day of the week have the greatest amount of search trends compared to any other time displayed. This information will enable our marketing team to deploy ads or relevant search data during peak hours to brute-force our internet dominance, or deploy during our least-search times of the day to establish a stronger basline of presence.

We can also use this data to find which weeks of the year have the highest traffic, as shown below:

![weeks](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/weeks.PNG)

Search Traffic tends to increase during the winter holiday period beginning in week 42, with a slight decrease in weeks 45 and 49, culminating in a high in week 51.

### Relate the Search Traffic to Stock Price Patterns
Principal Component Analysis (PCA) is a statistical technique used to accelerate machine learning algorithms when too many features exist. This is exectued to increase interpretability and minimize information loss.
1. Look for the following code:
    ```python
    from google.colab import files
    uploaded = files.upload()

    df_mercado_stock = pd.read_csv(
        "mercado_stock_price.csv", 
        index_col="date", 
        parse_dates=True, 
        infer_datetime_format=True
    )
    ```
Please ensure that you have the correct .CSV file uploaded.

As recent world events have shown, we cannot fully predict how the market will perform. With the recent COVID-19 pandemic, many companies and global financial markets received a shock in the first half of 2020. We will look at our market data during that time using the following code: ```first_half_2020 = mercado_stock_trends_df.loc["2020-01":"2020-06"]```

To show the relationship of our closing prices and search trends at that time, we will visualize the sliced data:

![twin](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/twin.PNG)

The data clearly shows a massive decrease in search trends and closing prices during May 2020. 

But why is this important? We will use these metrics to show our Stock Volatility versus the Search Trends over time:

![volatile](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/volatile.PNG)

This data can further be used to construct a correlation table using the following code: ```mercado_stock_trends_df[["Stock Volatility", "Lagged Search Trends", "Hourly Stock Return"]].corr()```

Based on the data given from the correlation, we see that a -14% relationship exists between the lagged search traffic and volatility, while the lagged search traffic and stock price boasts a 17% relationship.

### Create a Time Series Model with Prophet
According to [Facebook Prophet](https://facebook.github.io/prophet/), Prophet is "a forecasting procedure that provides completely automated forecasts". We will be using Prophet to create time-series models that analyzes and forecasts patterns across a broad spectrum of data types.

To use Prophet properly, we must:
1. Convert our DataFrame:
    ```python
    mercado_prophet_df = df_mercado_trends.reset_index()
    
    mercado_prophet_df.columns = ["ds", "y"]
    
    mercado_prophet_df = mercado_prophet_df.dropna()
    
    mercado_prophet_df
    ```
    
After producing our new DataFrame, we can create our model DataFrame using the ```Prophet()``` function. Fit this model to the previously created DataFrame.

To create prediction that will go as far as 2000 hours(approximately 80 days), run the following: <br>
    ```python
    future_mercado_trends = model_mercado_trends.make_future_dataframe(periods=2000, freq="H")
    ```

Make predictions for the data with the following: <br>
    ```python
    forecast_mercado_trends = model_mercado_trends.predict(future_mercado_trends)
    ```

This will provide us with the following visual forecast:

![forecast](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/forecast.PNG)

Viewing the data, we an se that the near-term forecast for the popularity of MercadoLibre will have a gradual decline until October 2020, upon when it will begin to increase.

### Forecast Revenue by Using Time Series Models
The Finance Division of MercadoLibre wants a forecast of the total sales for the next quarter. To do this we will:
1. Load the data:
    ```python
    from google.colab import files
    uploaded = files.upload()

    df_mercado_sales = pd.read_csv(
    "mercado_daily_revenue.csv", 
    index_col="date", 
    parse_dates=True, 
    infer_datetime_format=True
    )
    ```
2. Repeat the steps used for Prophet.
3. Produce a forecast:
![forecast2](https://github.com/antonmaliksi/FinTechModule11Challenge/blob/main/Readme%20Resources/forecast2.PNG)
4. Manipulate the data to find our expected sales, worst-case sales, and best-case sales:
    ```python
    mercado_sales_forecast_quarter = mercado_sales_prophet_forecast[["yhat", "yhat_lower", "yhat_upper"]].loc["2020-07-01":"2020-09-30"]
    
    mercado_sales_forecast_quarter = mercado_sales_forecast_quarter.rename(columns={"yhat":"Expected Sales", "yhat_lower":"Worst-Case", "yhat_upper":"Best-Case"})
    
    mercado_sales_forecast_quarter.sum()
    ```
5. Obersrve that on the forecast information, we can predict that MercadoLibre will have a positive return of approximately 969 sales, with a worst-case sales value of 887 and a best-case sales value of 1,051.

---

## Versioning History
All Github commits/pulls were conducted and verified by Growth Analyst Anton Maliksi.

---

## Contributors
Anton Maliksi was the sole contributor for this notebook, with assistance from Aarchit Malhotra.

---

## Licenses
No licenses were used for this project.