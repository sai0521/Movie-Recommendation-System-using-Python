# Movie Recommendation System

## Project Overview
This project implements a content-based movie recommendation system using Python and machine learning techniques. The system analyzes movie features like genres, plot overviews, keywords, cast members, and directors to recommend similar movies based on cosine similarity.

## Key Features
- **Data Integration**: Combines movie metadata from two datasets (`movies.csv` and `credits.csv`)
- **Feature Extraction**: Processes structured JSON data into usable features
- **Text Processing**: Implements stemming and vectorization for content analysis
- **Similarity Matching**: Uses cosine similarity to find related movies

## Workflow Steps

### 1. Data Loading and Preparation
```python
import numpy as np
import pandas as pd

# Load datasets
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

# Select relevant columns
movies = movies[['id','title','genres','overview','keywords']]
credits = credits[['title','cast','crew']]

# Merge datasets
movies = movies.merge(credits, on='title')
```

### 2. Data Preprocessing
- Parse JSON fields into Python lists
- Extract director from crew information
- Combine features into a unified "tags" column
- Clean text data by removing spaces and converting to lowercase

### 3. Text Processing
```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Apply stemming to tags
def stem_text(text):
    return ' '.join([ps.stem(word) for word in text.split()])
    
df['tags'] = df['tags'].apply(stem_text)
```

### 4. Vectorization and Similarity Calculation
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create document-term matrix
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)
```

### 5. Recommendation Function
```python
def recommend(movie):
    movie_idx = df[df['title'] == movie].index[0]
    distances = similarity[movie_idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(df.iloc[i[0]].title)
```

## Example Usage
```python
recommend('Avatar')

# Output:
# Aliens vs Predator: Requiem
# Aliens
# Falcon Rising
# Independence Day
# Titan A.E.
```

## Requirements
- Python 3.x
- Libraries: pandas, numpy, nltk, scikit-learn
- Datasets: movies.csv, credits.csv

## How It Works
1. The system processes movie data to create comprehensive "tags" for each movie
2. Text processing techniques (stemming) normalize the tags
3. Movies are converted to numerical vectors using CountVectorizer
4. Cosine similarity measures content similarity between movies
5. The recommendation function finds the most similar movies to the input title

This content-based approach recommends movies by finding films with similar textual content in their metadata, providing personalized suggestions without requiring user rating data.
