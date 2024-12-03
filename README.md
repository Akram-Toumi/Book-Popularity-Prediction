# Book Popularity Prediction

This machine learning project predicts the popularity of books based on various features such as review summary, review text, review helpfulness, price, and categories. The goal is to classify books into "Popular" or "Unpopular" categories using a Random Forest model.

## Dataset

The dataset used in this project is `books.csv`, which contains the following columns:

- `review/summary`: A brief summary of the book review.
- `review/text`: The full text of the book review.
- `review/helpfulness`: A string in the format 'good_reviews/total_reviews' representing the number of helpful reviews and the total number of reviews.
- `price`: The price of the book.
- `popularity`: The popularity of the book, classified as 'Popular' or 'Unpopular'.
- `authors`: The author(s) of the book.
- `categories`: The categories the book belongs to.

## Project Overview

### 1. Data Preprocessing

- **Feature Extraction**: The features extracted from the dataset include text data (`review/summary`, `review/text`), numerical data (`price`, `review/helpfulness`), and categorical data (`authors`, `categories`).
- **Splitting the 'review/helpfulness' Column**: The `review/helpfulness` column is split into two new columns, `good_reviews` and `total_reviews`, representing the number of helpful reviews and total reviews, respectively.
- **Price Scaling**: The price column is scaled using `StandardScaler` for normalization.
- **Popularity Classification**: The `popularity` column is converted into a binary classification target (`1` for popular, `0` for unpopular).
- **Text Processing**: `TfidfVectorizer` is used to convert `review/summary` and `review/text` into numerical features.
- **Categorical Data Encoding**: Authors are encoded using `LabelEncoder`, and categories are transformed into dummy variables.

### 2. Feature Engineering

- **Numerical Features**: The numerical features include scaled price, `good_reviews`, and `total_reviews`.
- **Text Features**: The text features are extracted using TF-IDF vectorization on the `review/summary` and `review/text`.
- **Categorical Features**: One-hot encoding is used to create dummy variables for the categories.

### 3. Model Training and Evaluation

- **Random Forest Classifier**: A Random Forest model is trained on the combined features, including numerical, text-based, and categorical features. The model is evaluated using accuracy on a test set.
