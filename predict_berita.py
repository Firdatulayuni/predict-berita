import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Inisialisasi NLTK Stopwords dan Stemmer Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk preprocessing teks
def preprocess(input_text):
    # Tambahkan fungsi preprocessing yang sudah kamu buat sebelumnya
    def remove_url(news):
        url = re.compile(r'https?://\S+|www\.S+')
        return url.sub(r'', news)

    def remove_html(news):
        html = re.compile(r'<.#?>')
        return html.sub(r'', news)

    def remove_emoji(news):
        emoji_pattern = re.compile("[" 
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"  
        "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', news)

    def remove_numbers(news):
        return re.sub(r'\d+', '', news)

    def remove_symbols(news):
        return re.sub(r'[^a-zA-Z0-9\s]', '', news)

    def case_folding(text):
        return text.lower() if isinstance(text, str) else text

    def tokenize(text):
        return text.split()

    def remove_stopwords(tokens):
        stop_words = stopwords.words('indonesian')
        return [word for word in tokens if word not in stop_words]

    def stem_text(tokens):
        return [stemmer.stem(word) for word in tokens]

    # Proses cleaning
    text = remove_url(input_text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_numbers(text)
    text = remove_symbols(text)
    text = case_folding(text)

    # Tokenization dan preprocessing lainnya
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    stemmed_tokens = stem_text(tokens)

    return ' '.join(stemmed_tokens)

# Judul aplikasi
st.title("Klasifikasi Berita")

# Load atau fit model TF-IDF
try:
    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
        tfidf_vect = pickle.load(tfidf_file)
except FileNotFoundError:
    # st.write("TF-IDF Vectorizer tidak ditemukan. Melakukan fitting...")
    
    # Load dataset untuk fitting
    data = pd.read_csv("C:/KULIAH/PPW/berita_cleaned.csv")  # Ganti dengan lokasi file yang sesuai
    X = data['stemmed_text']  # Ganti dengan nama kolom yang sesuai
    y = data['Kategori']  # Target yang akan diklasifikasikan
    
    # Split data menjadi training dan testing set (misalkan 80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan fit TF-IDF Vectorizer
    tfidf_vect = TfidfVectorizer()
    X_train_tfidf = tfidf_vect.fit_transform(X_train)

    # Simpan TF-IDF Vectorizer ke file .pkl
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vect, f)

# Load model PCA dan Logistic Regression
with open('pca_model.pkl', 'rb') as pca_file:
    pca_model = pickle.load(pca_file)

with open('logreg_pca_model.pkl', 'rb') as model_file:
    logreg_pca_model = pickle.load(model_file)

# Input Data Baru dari Pengguna
new_input = st.text_input("Masukkan teks berita baru:")

# Tombol untuk melakukan prediksi
if st.button("Prediksi Kategori"):
    if new_input:
        # Preprocess input
        processed_input = preprocess(new_input)

        # Transformasi dengan TF-IDF
        new_tfidf = tfidf_vect.transform([processed_input])

        # Transformasi dengan PCA
        new_tfidf_pca = pca_model.transform(new_tfidf.toarray())

        # Prediksi
        new_prediction = logreg_pca_model.predict(new_tfidf_pca)

        # Tampilkan hasil prediksi
        st.success("Prediksi untuk data baru: " + ("Kuliner" if new_prediction[0] == 0 else "Wisata"))
    else:
        st.warning("Silakan masukkan teks berita untuk mendapatkan prediksi.")