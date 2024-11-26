# Flipkart Review Classification - Sentiment Analysis

This project classifies Flipkart product reviews into positive or negative categories based on the rating and review content using machine learning.

## Steps and Features:

### X: **Data Preprocessing & Exploration**
- Loaded the data from a CSV file containing product reviews and ratings.
- Analyzed unique ratings and visualized the distribution of ratings using `countplot`.
- Converted ratings into binary labels: `1` for positive reviews (ratings >= 5), and `0` for negative reviews (ratings < 5).
  
### Y: **Text Preprocessing & Feature Extraction**
- Preprocessed the review text by removing punctuation, converting text to lowercase, and removing stopwords using NLTK.
- Used the `TfidfVectorizer` to convert text data into numerical features for machine learning, selecting the top 2500 features based on term frequency-inverse document frequency (TF-IDF).

### Z: **Modeling & Evaluation**
- Split the data into training and testing sets (33% test data).
- Trained a **Decision Tree Classifier** on the preprocessed text data to predict the sentiment label.
- Evaluated the model using accuracy score and confusion matrix to visualize the performance.

### WordCloud:
- A WordCloud was generated for positive reviews to visualize the most frequent words.

## Libraries Used:
- `Pandas`: For data manipulation and analysis.
- `Seaborn` & `Matplotlib`: For data visualization.
- `NLTK`: For text processing (stopwords removal, tokenization).
- `Scikit-learn`: For machine learning (TF-IDF, Decision Tree, model evaluation).
- `WordCloud`: For generating word clouds from review data.

## Code Output:
- **Accuracy Score**: Performance of the trained model on the training dataset.
- **Confusion Matrix**: Shows true positive, true negative, false positive, and false negative counts.

## Conclusion:
The model successfully predicts the sentiment of product reviews based on the ratings and text content, providing insights into customer opinions.
