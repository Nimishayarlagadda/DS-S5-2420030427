
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'Movie': [
        'Avengers',
        'Batman',
        'Superman',
        'Titanic',
        'Inception'
    ],
    'Description': [
        'action superhero marvel fight',
        'dark superhero action detective',
        'superhero flying action hero',
        'romance ship love drama',
        'dream science fiction thriller'
    ]
})

print("Movie Dataset:")
print(movies)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['Description'])

similarity = cosine_similarity(tfidf_matrix)

movie_name = 'Batman'

movie_index = movies[movies['Movie'] == movie_name].index[0]

scores = list(enumerate(similarity[movie_index]))

sorted_scores = sorted(
    scores,
    key=lambda x: x[1],
    reverse=True
)

print("\nRecommended Movies for Batman:")

for i in sorted_scores[1:4]:
    print(movies.iloc[i[0]]['Movie'])
