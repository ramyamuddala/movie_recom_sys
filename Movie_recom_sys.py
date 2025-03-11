import requests
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from textblob import TextBlob
from tabulate import tabulate  # Add this import

API_KEY = "f6d0bb99654ffdb2933158f13f6b28ff"
url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}"
response = requests.get(url).json()
movies = response['results']

for movie in movies[:3]:
    print(movie['title'], "-", movie['overview'])

# Load dataset
movies_df = pd.read_csv("tmdb.csv")

# Ensure 'overview' column exists
if 'overview' not in movies_df.columns:
    movies_df['overview'] = ""

# Vectorize movie descriptions
tfidf = TfidfVectorizer(stop_words='english', analyzer='word')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'].fillna(''))

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies based on similarity
def recommend_movies(title, df, sim_matrix):
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = [df.iloc[i[0]]['title'] for i in scores[1:11]]
    return top_movies

print(recommend_movies("Inception", movies_df, cosine_sim))

# Load data for collaborative filtering
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(movies_df[['genres', 'homepage', 'id']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Train model
model = SVD()
model.fit(trainset)

# Predict ratings
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# Analyze user mood
def analyze_mood(user_input):
    analysis = TextBlob(user_input)
    if analysis.sentiment.polarity > 0:
        return "Happy Mood - Try Comedy or Adventure Movies!"
    elif analysis.sentiment.polarity < 0:
        return "Sad Mood - How about Drama or Comfort Movies?"
    else:
        return "Neutral Mood - Maybe a Mystery or Sci-Fi movie?"

# Ask user for their preference
user_preference = input("Would you like to get movie recommendations based on your mood, by popularity, or similar to the selected movie? (Enter 'mood', 'popularity', or 'similar'): ")

if user_preference.lower() == 'mood':
    user_mood = input("How are you feeling today? ")
    print(analyze_mood(user_mood))
elif user_preference.lower() == 'popularity':
    # Suggest movies by popularity
    def suggest_movies_by_popularity(df):
        sorted_movies = df.sort_values(by='popularity', ascending=False)
        return sorted_movies['title'].tolist()[:10]

    print("Movies by popularity:", suggest_movies_by_popularity(movies_df))
elif user_preference.lower() == 'similar':
    selected_movie = input("Enter the title of the movie you want similar recommendations for: ")
    print("Movies similar to the selected movie:", recommend_movies(selected_movie, movies_df, cosine_sim))
    # Stop after suggesting similar movies
else:
    print("Invalid option. Please enter 'mood', 'popularity', or 'similar'.")

# Ensure the code runs without errors before taking user input
# print(analyze_mood("I feel down today"))

# Suggest movies based on genre
def suggest_movies_by_genre(genre, df):
    genre_movies = df[df['genres'].str.contains(genre, case=False, na=False)]
    return genre_movies['title'].tolist()[:10]

if user_preference.lower() not in ['similar', 'popularity']:  # Only ask for genre if not 'similar' or 'popularity'
    user_genre = input("Enter a genre you like: ")
    recommended_movies = suggest_movies_by_genre(user_genre, movies_df)
    print("Movies you might like:", recommended_movies)

# Display details of a selected movie
def get_movie_details(title, df):
    movie = df[df['title'] == title].iloc[0]
    details = [
        ["Title", movie['title']],
        ["Overview", movie['overview']],
        ["Genres", movie['genres']],
        ["Homepage", movie['homepage']]
    ]
    return details

# Continue to movie details after user selects preference
selected_movie = input("Select a movie from the recommendations: ")
movie_details = get_movie_details(selected_movie, movies_df)
print("Movie Details:")
print(tabulate(movie_details, headers=["Attribute", "Value"], tablefmt="grid"))

