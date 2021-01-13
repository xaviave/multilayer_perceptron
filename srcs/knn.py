from collections import Counter
import math


def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []

    for index, example in enumerate(data):
        distance = distance_fn(example[:-1], query)
        neighbor_distances_and_indices.append((distance, index))

    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)


def recommend_movies(movie_query, k_recommendations):
    raw_movies_data = []
    with open("movies.csv", "r") as md:
        next(md)

        for line in md.readlines():
            data_row = line.strip().split(",")
            raw_movies_data.append(data_row)

    movies_recommendation_data = []
    for row in raw_movies_data:
        data_row = list(map(float, row[2:]))
        movies_recommendation_data.append(data_row)

    recommendation_indices, _ = knn(
        movies_recommendation_data,
        movie_query,
        k=k_recommendations,
        distance_fn=euclidean_distance,
        choice_fn=lambda x: None,
    )

    movie_recommendations = []
    for _, index in recommendation_indices:
        movie_recommendations.append(raw_movies_data[index])

    return movie_recommendations


if __name__ == "__main__":
    the_post = [7.2, 1, 1, 0, 0, 0, 0, 1, 0]
    recommended_movies = recommend_movies(movie_query=the_post, k_recommendations=5)

    for recommendation in recommended_movies:
        print(recommendation[1])
