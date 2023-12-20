#importing necessary libraries and loading dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("C:\Stress Prediction\dataset\stress_1.csv")
print(data.head())
#print(data.info())

print(data.isnull().sum()) #checking if data contains any null value

import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.figure( figsize=(5,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Mapping numerical labels to textual labels
label_mapping = {0: "No Stress", 1: "Stress"}
data["label"] = data["label"].map(label_mapping)

# Selecting only the "text" and "label" columns
selected_columns = ["text", "label"]
data_subset = data[selected_columns]

# Printing the first few rows of the subsetted DataFrame
print(data_subset.head())
#Model training to use a cv.predict in counter vectorization method
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Example classifier, you can use any classifier here

# Assuming x and y are your text and label data
x = data["text"]
y = data["label"]

# Vectorize the text data using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(x)
print(X)
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a classifier (for example, Logistic Regression)
classifier = LogisticRegression()

# Train the classifier on the vectorized data
classifier.fit(xtrain, ytrain)

# Make predictions on the test data
predictions = classifier.predict(xtest)

# Calculate accuracy or percentages as needed

#Model training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain,xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.metrics import accuracy_score

# model.fit(xtrain, ytrain)
predictions = classifier.predict(xtest)

# Calculate accuracy percentage
accuracy = accuracy_score(ytest, predictions)

# Calculate the percentage of "Stress" and "No Stress" instances in the test data
total_instances = len(ytest)
stress_instances = sum(predictions == "Stress")
no_stress_instances = sum(predictions == "No Stress")

percentage_stress = (stress_instances / total_instances) * 100
percentage_no_stress = (no_stress_instances / total_instances) * 100

# Print the results
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Percentage of 'Stress' : {:.2f}%".format(percentage_stress))
print("Percentage of 'No Stress' : {:.2f}%".format(percentage_no_stress))

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = classifier.predict(data)
print(output)

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = classifier.predict(data)
print(output)