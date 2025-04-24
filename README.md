# moviegenreprediction
# 🎬 Movie Genre Classification Using YouTube Trailer Transcripts

This project is an end-to-end AI solution that classifies the genre(s) of a movie using only its trailer transcript. The goal is to explore how Natural Language Processing (NLP) can extract meaningful patterns from publicly available video transcripts — a novel and fun challenge!

---

## 🔍 Problem Statement

Can we predict the genre(s) of a movie using just the **text transcript of its trailer**?

Using YouTube trailer transcripts and MovieLens metadata, this project explores multiple machine learning approaches to solve this as a **multi-label classification problem**.

---

## 🧠 Technologies Used

- **Python**
- **Natural Language Processing**: spaCy, TF-IDF, CountVectorizer, NLTK
- **ML Algorithms**: Random Forest, Support Vector Machine (SVM), Naive Bayes
- **Deep Learning**: LSTM (Keras, TensorFlow)
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Data Sources**: MovieLens dataset, YouTube (via YouTubeTranscriptAPI)
- **Recommendation (Prototype)**: Collaborative filtering concept
- *(Planned)*: Streamlit for interactive genre prediction

---

## 🧪 Approach

1. **Data Collection**  
   - Fetched trailer transcripts via YouTubeTranscriptAPI  
   - Mapped YouTube videos to MovieLens metadata

2. **Preprocessing**  
   - Tokenization, Lemmatization (spaCy)  
   - Stopword removal, lowercasing  
   - TF-IDF and CountVectorizer embeddings

3. **Modeling**  
   - Trained and compared multiple models:  
     - Random Forest  
     - Support Vector Machine  
     - Naive Bayes  
     - LSTM Deep Learning

4. **Evaluation**  
   - F1-Score, Precision, Recall  
   - Confusion matrix visualizations  
   - Multi-label prediction analysis

5. **Prototype Recommendation System** *(WIP)*  
   - Filtered collaborative suggestions based on genre alignment

---

## 📊 Results

- LSTM model showed promising performance in predicting nuanced genre combinations.
- Rule-based filtering combined with NLP pre-processing improved baseline ML accuracy.
- Significant learning in using transcript-based data for content understanding.

*(Visuals and model performance charts to be added.)*

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/movie-genre-classifier.git
cd movie-genre-classifier

# Install dependencies
pip install -r requirements.txt

# Run notebooks for exploration
jupyter notebook notebooks/

# (Coming Soon) Streamlit App for interactive predictions
