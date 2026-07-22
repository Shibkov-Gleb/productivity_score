# 🤖 Telegram Productivity Predictor Bot

An end-to-end Machine Learning project that predicts daily productivity scores through a conversational Telegram bot interface.

This project demonstrates a complete ML lifecycle: from exploratory data analysis and model training to building an inference pipeline and deploying a dockerized production application.

## 🚀 Features

- **Conversational Interface:** Collects daily habit data (sleep, work hours, exercise, etc.) step-by-step via a Telegram Bot.
- **Robust ML Pipeline:** Uses a Decision Tree Regressor trained on the [Time Management and Productivity Insights](https://www.kaggle.com/datasets/hanaksoy/time-management-and-productivity-insights) dataset.
- **Production Architecture:** Strict separation between offline model training and online serving (inference).
- **Data Scaling Consistency:** Implements `.pkl` persistence for both the model and the `MinMaxScaler` to prevent train-serve skew.
- **Containerized:** Fully dockerized for seamless deployment.

## 🏗️ Project Architecture

The codebase is split by responsibility to ensure maintainability and reproducibility:

```text
my-productivity-bot/
├── data/
│   └── Time Management and Productivity Insights.csv  # Raw Kaggle dataset
├── notebooks/
│   └── eda_and_testing.ipynb        # Exploratory Data Analysis & initial modeling
├── ml/
│   ├── train.py                     # Script to train and persist the model & scaler
│   └── preprocess.py                # Reusable data cleaning and splitting logic
├── models/                          # Generated automatically by train.py
│   ├── productivity_model.pkl       # Serialized Random Forest model
│   ├── model_features.pkl           # Feature name order
│   └── scaler.pkl                   # Fitted MinMaxScaler
├── bot/
│   ├── main.py                      # Telegram bot and conversation handlers
│   └── inference.py                 # Interface between raw chat inputs and ML model
├── .env                             # Environment variables (not in version control)
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── README.md
```
