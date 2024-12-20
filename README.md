# Stock Prediction Using Machine Learning (Minor Project for Corizo)

## Project Overview

This project aims to predict the future stock prices of Tesla (TSLA) using machine learning techniques. By analyzing historical data, the project builds a model to forecast future stock prices, which can assist investors in making informed decisions. The dataset used contains daily stock prices of Tesla, which will be analyzed and used to train the machine learning model.

## Objective

- To develop a machine learning model to predict stock prices.
- To apply various machine learning techniques, including Linear Regression, Decision Trees, and Neural Networks, to stock price prediction.
- To evaluate the performance of different models and choose the best one for accurate stock predictions.

## Tools and Technologies

- **Python**: Programming language used for data analysis and model development.
- **Libraries**:
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  - **Matplotlib**: For data visualization.
  - **Scikit-learn**: For machine learning algorithms and model evaluation.
  - **TensorFlow/Keras**: For building neural network models.
- **Tesla Stock Dataset**: Historical stock data of Tesla (TSLA) obtained from sources like Yahoo Finance or Alpha Vantage.

## Setup Instructions

- To run this project in Jupyter Notebook, follow these steps:
- Download all dependencies (pandas, numpy, matplotlib,scikit-learn, tensorflow)
- Download the dataset and Source file and open it in **Jupyter Notebook**


## Dataset

- **Source**: Tesla stock data (TSLA) from Yahoo Finance
- **Features**: 
  - Open
  - High
  - Low
  - Close
  - Adjusted Close
- **Data Range**: From the earliest available date to the most recent data.
  
## Data Preprocessing

- **Handling Missing Values**: Removing or imputing missing data.
- **Normalization**: Scaling data using MinMaxScaler or StandardScaler.
- **Splitting Data**: Dividing the dataset into training and testing sets (e.g., 70% training, 30% testing).

## Machine Learning Models

1. **Linear Regression**: A basic model to predict stock prices based on linear relationships between features.

## Model Evaluation

- **Metrics**:
  - Mean Squared Error (MSE)
  - R-Squared (R²)

## Results and Analysis

- **Visualization**: Plotting predicted vs. actual stock prices using Matplotlib/Seaborn.
- **Insights**: Discussing the results and which model performs the best for stock price prediction.

## Conclusion

- **Summary**: Recap of the process, from data collection to model evaluation.
- **Future Work**: Exploring other models like LSTM for time-series forecasting, integrating news sentiment analysis, or adding more features such as social media data.

## References

- [Tesla Stock Data - Yahoo Finance](https://finance.yahoo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

