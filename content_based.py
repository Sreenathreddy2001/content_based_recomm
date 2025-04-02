from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances, linear_kernel
import ast
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
links=pd.read_csv("links.csv")
tags=pd.read_csv("tags.csv")
ratings=pd.read_csv("ratings.csv")
movies=pd.read_csv("movies.csv")
movies.drop_duplicates(inplace=True)
title_final=[]
for i in movies.title:
    title_final.append((i.split('(')[0]))
movies['title']=title_final
movies['genres']=movies.genres.str.replace('|',',')
genres=movies['genres'].tolist()
tokenized_sentences = [word_tokenize(genre.lower()) for genre in genres]
cleaned_tokenized_sentences = [[word for word in tokens if word != ','] for tokens in tokenized_sentences]
model = Word2Vec(sentences=cleaned_tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

def get_similar_genres(genres, top_n=5):
    genre_vectors = [model.wv[genre] for genre in genres if genre in model.wv]
    if not genre_vectors:
        return []

    avg_vector = sum(genre_vectors) / len(genre_vectors)
    similar_genres = model.wv.similar_by_vector(avg_vector, topn=top_n)
    return similar_genres
def recommend_movies_by_genre(user_genres, top_n=10):
    similar_genres = get_similar_genres(user_genres)
    recommended_movies = []
    for genre, _ in similar_genres:   
        count=0
        while count < len(cleaned_tokenized_sentences):
            if genre in cleaned_tokenized_sentences[count]:
                recommended_movies.append(count)
            count += 1
        
    return recommended_movies[0:top_n]
def recommend_movies(name_of_movie, top_n): 
    if name_of_movie == '':
        return str("Please enter the movie")
    genre_of_movie = str(movies[movies['title'] == name_of_movie]['genres'].iloc[0]).split(',')
    genre_of_movie = [element.lower() for element in genre_of_movie]
    list1 = recommend_movies_by_genre(genre_of_movie, top_n)

    
    related_movies = []
    for i in list1:
        new_row = movies.iloc[i:i+1].to_dict()
        related_movies.append(new_row)

    formatted_related_movies = []
    for j in related_movies:
        new_row = {key: values[next(iter(values))] for key, values in j.items()}
        formatted_related_movies.append(new_row)
    return formatted_related_movies

def recommend_movies(name_of_movie, top_n):
    if not name_of_movie.strip():
        return "Please enter a valid movie name."
    
    # Normalize title casing and remove spaces
    movies['title'] = movies['title'].str.strip().str.lower()
    name_of_movie = name_of_movie.strip().lower()

    # Filter movies by title
    filtered_movies = movies[movies['title'] == name_of_movie]

    # Check if movie exists in the dataset
    if filtered_movies.empty:
        return f"Movie '{name_of_movie}' not found in dataset."

    # Extract genres safely
    genre_of_movie = filtered_movies['genres'].iloc[0]  # No IndexError now
    genre_list = genre_of_movie.lower().replace(" ", "").split(',')

    # Get recommended movie indices
    recommended_indices = recommend_movies_by_genre(genre_list, top_n)

    # Convert indices to movie names
    recommended_movie_names = movies.loc[recommended_indices, 'title'].tolist()

    return recommended_movie_names

print(f"The Movie Recommendations based on Genres are: {recommend_movies('Flint', 5)}")
