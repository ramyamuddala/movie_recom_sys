# 🎬 Movie Recommendation System

A content-based movie recommendation system built using data from [TMDb (The Movie Database)](https://www.themoviedb.org/). This system suggests movies similar to a selected title based on various features like genres, keywords, cast, crew, and more.

---

## 🧠 Features

- Content-based filtering using movie metadata
- TMDb dataset integration for rich movie details
- Cosine similarity to find movie similarities
- Cleaned and preprocessed data for efficient performance

---

## 📁 Dataset

The project uses datasets derived from TMDb, including:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

These files provide details such as:
- Movie titles, overviews, genres
- Cast and crew data
- Keywords related to the movies
- Ratings and popularity metrics

---

## 🛠️ Tech Stack

- **Python** (pandas, numpy, sklearn)
- **Scikit-learn** for vectorization and similarity calculation
- **TMDb** dataset
- (Optional) **Streamlit** or **Flask** for web interface

---

## 📊 How It Works

1. **Data Preprocessing**  
   Merges `movies_metadata`, `credits`, and `keywords` datasets and extracts relevant features.

2. **Feature Engineering**  
   Constructs a combined text-based feature (e.g., genres + cast + director + keywords).

3. **Vectorization**  
   Applies CountVectorizer or TF-IDF to convert text into numerical vectors.

4. **Similarity Calculation**  
   Uses Cosine Similarity to find similar movies.

5. **Recommendation Output**  
   Returns the top N similar movies based on similarity scores.

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
