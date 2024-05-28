from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import csv

app = Flask(__name__)

# Load the trained Naive Bayes model
model_filename = 'naive_bayes_model.joblib'
nb_model = joblib.load(model_filename)

# Load the fitted vectorizer
vectorizer_filename = 'vectorizer.joblib'
vectorizer = joblib.load(vectorizer_filename)

# Function to analyze sentiment
def analyze_sentiment(user_input):
    if not user_input:
        return "Please enter a sentence."

    # Convert input to lowercase for case-insensitive matching
    user_input_lower = user_input.lower()

    # Check for specific words to determine sentiment
    if any(word in user_input_lower for word in ['Kind','Friendly','Nice','Respectful',
'Tolerant','Inclusive','Accepting',
'Polite','Positive','good']):
        return user_input, 'Not-Hate'
    elif any(word in user_input_lower for word in ['fuck','fucking','bad', 'angry', 'harsh','shit','nigga','asshole','Hate','Racist','Discrimination','Bigotry','Prejudice','Intolerance','Stereotyping','Violence','Slur','Corrupt''Dictator','Traitor','Criminal','Thief','Liar','Tyrant','Despot','Puppet','Sellout']):
        return user_input, 'Hate'
    elif any(word in user_input_lower for word in ['poor','normal','']):
        return user_input, 'Neutral'
    
    # If none of the specific words are found, use the Naive Bayes model
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = nb_model.predict(user_input_vectorized)[0]
    sentiment_labels = {0: 'Neutral', 1: 'Not-Hate', 2: 'Hate'}
    predicted_sentiment = sentiment_labels[prediction]

    return user_input, predicted_sentiment
 # Return both input and sentiment

# Function to save data to CSV file
def save_to_csv(sentence, result):
    with open('hate_speech_analysis.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([sentence, result])

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for sentiment analysis
@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    user_input = request.form['sentence']
    input_sentence, result = analyze_sentiment(user_input)
    save_to_csv(input_sentence, result)  # Save input and sentiment to CSV
    return jsonify({"sentiment": result})

from flask import send_from_directory
from flask import send_from_directory



if __name__ == '__main__':
    app.run(debug=True)






