# Importing necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample movie data
movies = {
    'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Fight Club'],
    'Genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama'],
    'Rating': [9.3, 9.2, 9.0, 8.9, 8.8]
}

# Create DataFrame
df = pd.DataFrame(movies)

# Function to create a string with movie features
def combine_features(row):
    return row['Genre']

# Apply function to create new 'combined_features' column
df['combined_features'] = df.apply(combine_features, axis=1)

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit and transform 'combined_features' column
count_matrix = cv.fit_transform(df['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Function to get movie recommendations based on user preferences
def get_recommendations(movie_title):
    # Get movie index
    movie_index = df[df['Title'] == movie_title].index[0]
    
    # Get pairwise similarity scores with other movies
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    
    # Sort similar movies based on similarity scores
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself from recommendations
    sorted_similar_movies = sorted_similar_movies[1:]
    
    # Print top 3 recommended movies
    print("Top 3 Recommended Movies for", movie_title + ":")
    for i in range(3):
        print(df.iloc[sorted_similar_movies[i][0]]['Title'])

# Example usage
get_recommendations('The Shawshank Redemption')
