from flask import Flask, request, jsonify, render_template
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

app = Flask(__name__)

train_data = pd.read_csv(r"C:\Users\amrit\OneDrive\Desktop\new new\Train.csv")
test_data = pd.read_csv(r"C:\Users\amrit\OneDrive\Desktop\new new\Test.csv")

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
   
    # Removing stopwords, including 'no' and 'not'
    stop_words = set(stopwords.words('english'))
    stop_words.discard('no')
    stop_words.discard('not')
    tokens = [word for word in tokens if word.lower() not in stop_words]
   
    return tokens


def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]


def pos_tagging(text):
    return pos_tag(word_tokenize(text))


def named_entity_recognition(text):
    return ne_chunk(pos_tag(word_tokenize(text)))

def majority_vote(predictions):
    vote_counts = Counter(predictions)
    return vote_counts.most_common(1)[0][0]

# Preprocess the titles in training data
train_data['Title'] = train_data['Title'].apply(preprocess_text)

# Check if there are any non-empty tokens after preprocessing
train_data['Title_length'] = train_data['Title'].apply(len)
non_empty_titles = train_data[train_data['Title_length'] > 0]

if len(non_empty_titles) == 0:
    raise ValueError("All titles are empty after preprocessing. Please check your preprocessing steps.")

# Vectorization
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(non_empty_titles['Title'].apply(lambda x: ' '.join(x)))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_counts, train_data['Label'], test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Stochastic Gradient Descent": SGDClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train all classifiers
trained_classifiers = {}
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    trained_classifiers[name] = classifier

# Predictions from all classifiers
@app.route('/predict', methods=['POST'])
def predict_label_combined():
    data = request.json
    input_text = data['text']
    
    input_text_processed = preprocess_text(input_text)
    input_text_vectorized = vectorizer.transform([' '.join(input_text_processed)])
   
    predictions = {}
    for name, classifier in trained_classifiers.items():
        prediction = classifier.predict(input_text_vectorized)
        predictions[name] = prediction[0]
   
    final_prediction = majority_vote(predictions.values())
    
    return jsonify({'prediction': 'True' if final_prediction == 1 else 'False'})

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Classification</title>
    </head>
    <body>
        <h1>Text Classification</h1>
        <form id="textForm">
            <label for="textInput">Enter Text:</label><br>
            <textarea id="textInput" name="textInput" rows="4" cols="50"></textarea><br>
            <button type="button" onclick="predictLabel()">Predict Label</button>
            <div id="loading" style="display:none;">Loading...</div>
        </form>
        <p id="predictionResult"></p>

        <script>
            function predictLabel() {
                var text = document.getElementById("textInput").value;

                // Show loading symbol
                document.getElementById("loading").style.display = "block";

                // AJAX request to backend for prediction
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "/predict", true);
                xhr.setRequestHeader("Content-Type", "application/json");
                xhr.onreadystatechange = function () {
                    if (xhr.readyState === 4 && xhr.status === 200) {
                        // Hide loading symbol
                        document.getElementById("loading").style.display = "none";
                        
                        var prediction = xhr.responseText;
                        document.getElementById("predictionResult").innerText = "Predicted Label: " + prediction;
                    }
                };
                xhr.send(JSON.stringify({ text: text }));
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
