# Twitter Sentiment Analysis using Machine Learning

## Overview
This project focuses on performing sentiment and emotion analysis on tweets regarding the **Russia-Ukraine conflict** using machine learning algorithms. The goal is to classify tweets into positive, neutral, or negative sentiments, as well as emotions like anger, joy, optimism, and sadness.

## Dataset
The dataset used for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows) and contains tweets related to the Russia-Ukraine conflict. The dataset contains the following features:
- **Text**: The tweet's text.
- **Location**: The location from where the tweet was sent.
- **Sentiment**: Sentiment of the tweet (positive, negative, or neutral).
- **Emotion**: Emotions expressed in the tweet (anger, joy, optimism, sadness).

## Project Structure
- **Data Preprocessing**: Cleaning the data by removing noise like URLs, emojis, and stop words. Lemmatization and stemming are also applied to standardize the text.
- **Feature Extraction**: Applying **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
- **Model Building**: Implementing the following machine learning algorithms:
  - Bernoulli Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)

- **Model Evaluation**: Models are evaluated based on accuracy, precision, recall, F1 score, and confusion matrix.

## Algorithms and Accuracy
- **SVM**: 93.70% accuracy
- **Logistic Regression**: 93% accuracy
- **Random Forest**: 91% accuracy
- **Decision Tree**: 88% accuracy
- **Bernoulli Naive Bayes**: 87% accuracy
- **KNN**: 80% accuracy

## Key Findings
- The **SVM** classifier outperformed other algorithms with the highest accuracy of **93.70%**.
- Logistic Regression and Random Forest also achieved high accuracy, but KNN performed the worst in this case.

## How to Run

### Prerequisites
- Python 3.x
- Jupyter Notebook or a similar environment
- Required Libraries: Install them using the following command:
  ```bash
  pip install -r requirements.txt


# Steps to Execute
# Clone the repository:

git clone https://github.com/yourusername/twitter-sentiment-analysis.git
# Navigate to the project directory:

cd twitter-sentiment-analysis
# Run the Jupyter Notebook to execute the project:

jupyter notebook Twitter_Sentiment_Analysis.ipynb
# Future Work
- Use deep learning models like LSTM for better text classification.
- Apply real-time sentiment analysis on live Twitter feeds using the Twitter API.
- Explore more emotion categories for better insights into public reactions.
# Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your changes.
