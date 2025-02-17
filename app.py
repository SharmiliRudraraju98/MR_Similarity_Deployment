from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

app = Flask(__name__)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Google Sheets setup
# scope = [
#     'https://www.googleapis.com/auth/spreadsheets',
#     'https://www.googleapis.com/auth/drive'
# ]

# try:
#     credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
#     gc = gspread.authorize(credentials)
#     spreadsheet = gc.open('mr_similarity').sheet1
# except Exception as e:
#     print(f"Error setting up Google Sheets: {e}")

def preprocess_text(text):
    """Basic preprocessing"""
    return text.lower().strip()

def hybrid_similarity(text1, text2):
    """Compute similarity using TF-IDF and Levenshtein"""
    # Preprocess texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    # TF-IDF similarity
    tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
    semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Levenshtein similarity
    edit_distance_score = ratio(text1_clean, text2_clean)
    
    # Weighted combination
    final_score = (semantic_score * 0.7) + (edit_distance_score * 0.3)
    
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
        
        similarity_percentage = hybrid_similarity(memory1, memory2)
        
        # spreadsheet.append_row([participant_code, similarity_percentage])
        
        return jsonify({'similarity': similarity_percentage})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))