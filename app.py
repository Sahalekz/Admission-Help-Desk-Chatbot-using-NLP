import json
from flask import Flask, request, jsonify, render_template, send_from_directory
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pyttsx3

# Download NLTK resources (only needed once)
nltk.download('punkt')

app = Flask(__name__)

# Load FAQs from JSON file
def load_faqs(file_path='admission.json'):
    with open(file_path, 'r') as file:
        faqs = json.load(file)["faqs"]
    return faqs

# Tokenize and preprocess the questions
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

# Correct spelling mistakes in user input
def correct_spelling(text):
    corrected_text = str(TextBlob(text).correct())
    return corrected_text

# Find the best matching question
def find_best_match(user_question, faqs):
    user_question = preprocess_text(user_question)
    questions = [faq['question'] for faq in faqs]
    preprocessed_questions = [preprocess_text(q) for q in questions]
    
    vectorizer = TfidfVectorizer().fit_transform([user_question] + preprocessed_questions)
    vectors = vectorizer.toarray()
    
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_index = cosine_similarities.argmax()
    
    return faqs[best_match_index]

# Main chatbot function with context handling
user_context = {}

def chatbot(user_question, user_id, faqs):
    if user_id not in user_context:
        user_context[user_id] = {}

    if "awaiting_department" in user_context[user_id]:
        department = user_question
        user_context[user_id].pop("awaiting_department")
        return f"The fees for the {department} department is $xxxx."

    corrected_question = correct_spelling(user_question)
    best_match_faq = find_best_match(corrected_question, faqs)
    
    if "fees" in corrected_question.lower():
        user_context[user_id]["awaiting_department"] = True
        return "Which department are you interested in? Here are the available departments: Computer Science(DataScience), Computer Science(CyberSec), Computer Science(AI&ML), Social Work."

    response = best_match_faq['answer']
    if 'image' in best_match_faq:
        response += f"\n[image]{best_match_faq['image']}[/image]"
    
    return response

# Load the FAQs once
faqs = load_faqs()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    user_question = data.get("message")
    user_id = data.get("user_id", "default_user")
    response = chatbot(user_question, user_id, faqs)
    return jsonify({"response": response})

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route("/voice_input", methods=["POST"])
def voice_input():
    user_id = request.json.get("user_id", "default_user")
    user_question = recognize_speech_from_mic()
    response = chatbot(user_question, user_id, faqs)
    speak_text(response)
    return jsonify({"response": response})

@app.route('/static/images/<path:filename>')
def send_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == "__main__":
    app.run(debug=True)
