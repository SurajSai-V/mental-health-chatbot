from flask import Flask, request, jsonify, render_template
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def get_response(message, sentiment):
    if sentiment > 0.2:
        return "I'm glad you're feeling positive! How can I support you today?"
    elif sentiment < -0.2:
        return "I'm sorry you're going through a tough time. Would you like to talk about it?"
    else:
        return "I'm here to listen. Tell me more about how you're feeling."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    sentiment = TextBlob(message).sentiment.polarity
    response = get_response(message, sentiment)
    return jsonify({
        'response': response,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)