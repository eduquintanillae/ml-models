# from utils.nlp import cosine_similarity
# from utils.deep_learning import random_tensor
from typing import Dict, List, Tuple, NamedTuple
import tqdm
from utils.linear_algebra import dot
import csv
import re
from collections import Counter, defaultdict
import random
from utils.working_with_data import pca, transform

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests)

def most_popular_new_interests(
        user_interests: List[str],
        max_results: int = 5) -> List[Tuple[str, int]]:
    suggestions = [(interest, frequency)
                   for interest, frequency in popular_interests.most_common()
                   if interest not in user_interests]
    return suggestions[:max_results]

unique_interests = sorted({interest
                           for user_interests in users_interests
                           for interest in user_interests})

def make_user_interest_vector(user_interests: List[str]) -> List[int]:
    """
    Given a list ofinterests, produce a vector whose ith element is 1
    if unique_interests[i] is in the list, 0 otherwise
    """
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

user_interest_vectors = [make_user_interest_vector(user_interests)
                         for user_interests in users_interests]


user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_vectors]
                     for interest_vector_i in user_interest_vectors]

def most_similar_users_to(user_id: int) -> List[Tuple[int, float]]:
    pairs = [(other_user_id, similarity)                      # Find other
             for other_user_id, similarity in                 # users with
                enumerate(user_similarities[user_id])         # nonzero
             if user_id != other_user_id and similarity > 0]  # similarity.

    return sorted(pairs,                                      # Sort them
                  key=lambda pair: pair[-1],                  # most similar
                  reverse=True)                               # first.


most_similar_to_zero = most_similar_users_to(0)
user, score = most_similar_to_zero[0]
user, score = most_similar_to_zero[1]

def user_based_suggestions(user_id: int,
                           include_current_interests: bool = False):
    suggestions: Dict[str, float] = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
            
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

ubs0 = user_based_suggestions(0)
interest, score = ubs0[0]
interest, score = ubs0[1]

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_vectors]
                        for j, _ in enumerate(unique_interests)]

interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

def most_similar_interests_to(interest_id: int):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)


msit0 = most_similar_interests_to(0)

def item_based_suggestions(user_id: int,
                           include_current_interests: bool = False):
    # Add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_vectors[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # Sort them by weight
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

[('MapReduce', 1.861807319565799),
 ('Postgres', 1.3164965809277263),
 ('MongoDB', 1.3164965809277263),
 ('NoSQL', 1.2844570503761732),
 ('programming languages', 0.5773502691896258),
 ('MySQL', 0.5773502691896258),
 ('Haskell', 0.5773502691896258),
 ('databases', 0.5773502691896258),
 ('neural networks', 0.4082482904638631),
 ('deep learning', 0.4082482904638631),
 ('C++', 0.4082482904638631),
 ('artificial intelligence', 0.4082482904638631),
 ('Python', 0.2886751345948129),
 ('R', 0.2886751345948129)]

ibs0 = item_based_suggestions(0)
def main():
    MOVIES = "u.item"
    RATINGS = "u.data"
    
    class Rating(NamedTuple):
        user_id: str
        movie_id: str
        rating: float
    
    with open(MOVIES, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter="|")
        movies = {movie_id: title for movie_id, title, *_ in reader}
    
    with open(RATINGS, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter="\t")
        ratings = [Rating(user_id, movie_id, float(rating))
                   for user_id, movie_id, rating, _ in reader]    
    
    star_wars_ratings = {movie_id: []
                         for movie_id, title in movies.items()
                         if re.search("Star Wars|Empire Strikes|Jedi", title)}
    
    for rating in ratings:
        if rating.movie_id in star_wars_ratings:
            star_wars_ratings[rating.movie_id].append(rating.rating)
    
    avg_ratings = [(sum(title_ratings) / len(title_ratings), movie_id)
                   for movie_id, title_ratings in star_wars_ratings.items()]
    
    for avg_rating, movie_id in sorted(avg_ratings, reverse=True):
        print(f"{avg_rating:.2f} {movies[movie_id]}")
    
    random.seed(0)
    random.shuffle(ratings)
    
    split1 = int(len(ratings) * 0.7)
    split2 = int(len(ratings) * 0.85)
    
    train = ratings[:split1]
    validation = ratings[split1:split2]
    test = ratings[split2:]
    
    avg_rating = sum(rating.rating for rating in train) / len(train)
    baseline_error = sum((rating.rating - avg_rating) ** 2
                         for rating in test) / len(test)
    
    EMBEDDING_DIM = 2
    user_ids = {rating.user_id for rating in ratings}
    movie_ids = {rating.movie_id for rating in ratings}
    user_vectors = {user_id: random_tensor(EMBEDDING_DIM)
                    for user_id in user_ids}
    movie_vectors = {movie_id: random_tensor(EMBEDDING_DIM)
                     for movie_id in movie_ids}
    
    def loop(dataset: List[Rating],
             learning_rate: float = None) -> None:
        with tqdm.tqdm(dataset) as t:
            loss = 0.0
            for i, rating in enumerate(t):
                movie_vector = movie_vectors[rating.movie_id]
                user_vector = user_vectors[rating.user_id]
                predicted = dot(user_vector, movie_vector)
                error = predicted - rating.rating
                loss += error ** 2
    
                if learning_rate is not None:
                    user_gradient = [error * m_j for m_j in movie_vector]
                    movie_gradient = [error * u_j for u_j in user_vector]
                    
                    for j in range(EMBEDDING_DIM):
                        user_vector[j] -= learning_rate * user_gradient[j]
                        movie_vector[j] -= learning_rate * movie_gradient[j]
    
                t.set_description(f"avg loss: {loss / (i + 1)}")
    
    learning_rate = 0.05
    for epoch in range(20):
        learning_rate *= 0.9
        print(epoch, learning_rate)
        loop(train, learning_rate=learning_rate)
        loop(validation)
    loop(test)
        
    original_vectors = [vector for vector in movie_vectors.values()]
    components = pca(original_vectors, 2)
    
    ratings_by_movie = defaultdict(list)
    for rating in ratings:
        ratings_by_movie[rating.movie_id].append(rating.rating)
    
    vectors = [
        (movie_id,
         sum(ratings_by_movie[movie_id]) / len(ratings_by_movie[movie_id]),
         movies[movie_id],
         vector)
        for movie_id, vector in zip(movie_vectors.keys(),
                                    transform(original_vectors, components))
    ]
    
    print(sorted(vectors, key=lambda v: v[-1][0])[:25])
    print(sorted(vectors, key=lambda v: v[-1][0])[-25:])
    
if __name__ == "__main__": 
    main()