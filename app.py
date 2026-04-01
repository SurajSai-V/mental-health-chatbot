from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import firebase_admin
from firebase_admin import credentials, db, auth as firebase_auth
import datetime
import uuid
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = Flask(__name__)

try:
    firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
    if firebase_creds_json:
        firebase_creds = json.loads(firebase_creds_json)
        cred = credentials.Certificate(firebase_creds)
    else:
        cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ.get('FIREBASE_DATABASE_URL', 'https://mental-health-chatbot-4db12-default-rtdb.asia-southeast1.firebasedatabase.app')
    })
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase initialization error: {e}")

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq initialized successfully")
except Exception as e:
    print(f"Groq initialization error: {e}")

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are MoodBoost, a compassionate mental health companion. Follow these rules strictly:

RESPONSE STYLE:
- For casual talk or greetings: reply naturally and conversationally in 1-2 sentences. No bullet points.
- For emotional sharing (sad, anxious, stressed): respond with empathy first, then 2-3 short bullet points with practical suggestions if needed.
- For questions asking for solutions or advice: give a clear, structured response with numbered steps or bullet points. Make it easy to read — no dense paragraphs.
- Never write walls of text. Keep responses concise and scannable.
- Use line breaks between points.
- Be warm, human, and genuine — not robotic.

SAFETY:
- Never give medical advice.
- If someone is in danger, always suggest contacting a professional or crisis helpline.
- You are not a therapist but you care deeply."""
}


def verify_user(request):
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return None
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded['uid']
    except Exception as e:
        print(f"Token verification error: {e}")
        return None


def get_session_history(uid, session_id):
    ref = db.reference(f'users/{uid}/sessions/{session_id}/messages')
    data = ref.get()
    messages = [SYSTEM_PROMPT]
    if data:
        for key in sorted(data.keys()):
            entry = data[key]
            messages.append({"role": entry['role'], "content": entry['content']})
    return messages


def save_session_message(uid, session_id, role, content):
    db.reference(f'users/{uid}/sessions/{session_id}/messages').push({
        'role': role,
        'content': content,
        'timestamp': datetime.datetime.now().isoformat()
    })


def create_session(uid, session_id, title):
    db.reference(f'users/{uid}/sessions/{session_id}').update({
        'title': title[:40],
        'timestamp': datetime.datetime.now().isoformat()
    })


def get_ai_response(uid, message, sentiment, session_id):
    if sentiment > 0.2:
        mood = "The user seems to be in a positive mood."
    elif sentiment < -0.2:
        mood = "The user seems to be feeling negative or distressed."
    else:
        mood = "The user seems to be in a neutral mood."

    user_content = f"{message} (Note for AI only: {mood})"
    save_session_message(uid, session_id, 'user', user_content)
    history = get_session_history(uid, session_id)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        max_tokens=150
    )

    reply = response.choices[0].message.content
    save_session_message(uid, session_id, 'assistant', reply)
    return reply


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sessions', methods=['GET'])
def get_sessions():
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        ref = db.reference(f'users/{uid}/sessions')
        data = ref.get()
        if not data:
            return jsonify([])
        sessions = []
        for key, val in data.items():
            if not isinstance(val, dict) or 'title' not in val:
                continue
            sessions.append({
                'id': key,
                'title': val.get('title', 'Untitled'),
                'timestamp': val.get('timestamp', '')
            })
        sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(sessions[:30])
    except Exception as e:
        print("ERROR /sessions:", e)
        return jsonify([]), 500


@app.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        ref = db.reference(f'users/{uid}/sessions/{session_id}/messages')
        data = ref.get()
        if not data:
            return jsonify([])
        messages = []
        for key in sorted(data.keys()):
            entry = data[key]
            content = entry['content']
            if entry['role'] == 'user' and ' (Note for AI only:' in content:
                content = content[:content.index(' (Note for AI only:')]
            messages.append({
                'role': entry['role'],
                'content': content,
                'timestamp': entry.get('timestamp', '')
            })
        return jsonify(messages)
    except Exception as e:
        print("ERROR /sessions/<id>/messages:", e)
        return jsonify([]), 500


@app.route('/chat', methods=['POST'])
def chat():
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', '')
        is_new_session = data.get('is_new_session', False)

        if not session_id:
            session_id = str(uuid.uuid4())
            is_new_session = True

        if is_new_session:
            create_session(uid, session_id, message)

        sentiment = TextBlob(message).sentiment.polarity
        response = get_ai_response(uid, message, sentiment, session_id)

        db.reference(f'users/{uid}/conversations').push({
            'user_message': message,
            'sentiment_score': sentiment,
            'bot_response': response,
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat()
        })

        return jsonify({
            'response': response,
            'sentiment': sentiment,
            'session_id': session_id
        })
    except Exception as e:
        print("ERROR /chat:", e)
        return jsonify({'response': 'Error: ' + str(e)}), 500


@app.route('/clear/<session_id>', methods=['POST'])
def clear_session(session_id):
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        db.reference(f'users/{uid}/sessions/{session_id}/messages').delete()
        return jsonify({'message': 'Session cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete/<session_id>', methods=['POST'])
def delete_session(session_id):
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        db.reference(f'users/{uid}/sessions/{session_id}').delete()
        return jsonify({'message': 'Session deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cbt', methods=['POST'])
def cbt():
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json
        mood = data.get('mood', '')
        messages = [
            {
                "role": "system",
                "content": """You are a CBT (Cognitive Behavioral Therapy) counselor.
Give practical, evidence-based CBT guidance for the user's emotional state.
Structure your response with:
1. A brief validation of their feeling (1 sentence)
2. One key CBT technique relevant to their situation
3. A short actionable exercise they can do right now
Keep it warm, practical, and under 150 words."""
            },
            {"role": "user", "content": f"I am feeling: {mood}"}
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=200
        )
        return jsonify({'advice': response.choices[0].message.content})
    except Exception as e:
        print("ERROR /cbt:", e)
        return jsonify({'advice': 'Error: ' + str(e)}), 500


@app.route('/analysis/stats', methods=['GET'])
def analysis_stats():
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        ref = db.reference(f'users/{uid}/conversations')
        data = ref.get()
        if not data:
            return jsonify({'total': 0, 'positive': 0, 'negative': 0, 'trend': []})
        entries = list(data.values())
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
        recent = [e for e in entries if e.get('timestamp', '') >= cutoff]
        total = len(recent)
        positive = sum(1 for e in recent if e.get('sentiment_score', 0) > 0.2)
        negative = sum(1 for e in recent if e.get('sentiment_score', 0) < -0.2)
        from collections import defaultdict
        daily = defaultdict(list)
        for e in recent:
            day = e.get('timestamp', '')[:10]
            daily[day].append(e.get('sentiment_score', 0))
        trend = [
            {'date': day, 'score': round(sum(scores) / len(scores), 3)}
            for day, scores in sorted(daily.items())
        ]
        return jsonify({'total': total, 'positive': positive, 'negative': negative, 'trend': trend})
    except Exception as e:
        print("ERROR /analysis/stats:", e)
        return jsonify({'total': 0, 'positive': 0, 'negative': 0, 'trend': []}), 500


@app.route('/analysis/summary', methods=['POST'])
def analysis_summary():
    uid = verify_user(request)
    if not uid:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        ref = db.reference(f'users/{uid}/conversations')
        data = ref.get()
        if not data:
            return jsonify({'summary': 'No conversation data yet. Start chatting to see your mood analysis!'})
        entries = list(data.values())
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
        recent = [e for e in entries if e.get('timestamp', '') >= cutoff]
        if not recent:
            return jsonify({'summary': 'No conversations in the last 30 days.'})
        scores = [e.get('sentiment_score', 0) for e in recent]
        avg = sum(scores) / len(scores)
        messages_summary = ' | '.join([e.get('user_message', '')[:60] for e in recent[-10:]])
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a compassionate mental health analyst. Analyze the user's recent chat data and provide a warm, insightful 3-4 sentence summary of their emotional patterns over the past month. Be encouraging and constructive."
            },
            {
                "role": "user",
                "content": f"My recent messages: {messages_summary}\nAverage sentiment score: {avg:.2f} (range -1 to 1, negative is distressed, positive is happy)"
            }
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=prompt_messages,
            max_tokens=200
        )
        return jsonify({'summary': response.choices[0].message.content})
    except Exception as e:
        print("ERROR /analysis/summary:", e)
        return jsonify({'summary': 'Error generating summary.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)