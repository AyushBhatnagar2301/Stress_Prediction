# Stress_Prediction

Data Preprocessing:

We loaded a CSV file using pandas.
Checked for null values.
Applied text cleaning by removing special characters, links, punctuation, and stopwords. You also performed stemming.
Text Visualization:

Created a word cloud to visualize the most frequent words in the cleaned text.
Label Mapping:

Mapped numerical labels to textual labels (0: "No Stress", 1: "Stress").
Feature Vectorization:

Used CountVectorizer to convert the text data into a bag-of-words representation.
Model Training:

Split the data into training and testing sets.
Trained a Logistic Regression classifier using the training data.
Evaluated the model's performance on the test set, calculating accuracy and percentages of "Stress" and "No Stress" instances.
User Input and Prediction:

Prompted the user to input text.
Used the trained model to predict whether the input text indicates "Stress" or "No Stress."
