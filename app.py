from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from Levenshtein import ratio
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

app = Flask(__name__)

# Load a better model for sentence-level similarity
model = SentenceTransformer('sentence-t5-large')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Google Sheets setup
scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

try:
    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open('mr_similarity').sheet1
except Exception as e:
    print(f"Error setting up Google Sheets: {e}")

def preprocess_text(text):
    """Preprocess text by lowercasing and lemmatizing words."""
    words = text.lower().strip().split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def hybrid_similarity(text1, text2):
    """Compute a combined similarity score using SBERT + Levenshtein ratio."""
    # Preprocess input texts
    text1, text2 = preprocess_text(text1), preprocess_text(text2)
    
    # Compute SBERT similarity
    embedding1 = model.encode(text1, normalize_embeddings=True)
    embedding2 = model.encode(text2, normalize_embeddings=True)
    semantic_score = util.cos_sim(embedding1, embedding2).item()
    
    # Compute character-based similarity (Levenshtein distance)
    edit_distance_score = ratio(text1, text2)  # 0 to 1 scale
    
    # Weighted combination (adjust weights as needed)
    final_score = (semantic_score * 0.77) + (edit_distance_score * 0.23)
    
    return round(final_score * 100, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        memory1 = data['memory1']
        memory2 = data['memory2']
        participant_code = data['participant_code']
        email = data['email']
        
        # Compute hybrid similarity
        similarity_percentage = hybrid_similarity(memory1, memory2)
        
        # Save only necessary data to Google Sheets
        spreadsheet.append_row([participant_code, email, similarity_percentage])
        
        return jsonify({'similarity': similarity_percentage})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))