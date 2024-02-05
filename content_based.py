import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def custom_tokenizer(text):
    return text.split("|")


def main():
    # | Read Data |
    movies = pd.read_csv("data/ml-latest-small/ml-latest-small/movies.csv")
    tags = pd.read_csv("data/ml-latest-small/ml-latest-small/tags.csv")

    # | Split year from title |
    movies["year"] = movies["title"].str.extract("(\d{4})")
    movies["title"] = movies["title"].str.replace("(\ \(\d{4}\))", "")
    movies["year"] = movies["year"].fillna("2002")

    # | Use Tag |
    movie_group = tags.groupby("movieId")["tag"].apply("|".join).reset_index()
    movies = pd.merge(movies, movie_group, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("unkown")

    # | Make content |
    movies["content"] = movies["year"] + "|" + movies["genres"] + "|" + movies["tag"]

    # | Compute |
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    one_hot_matrix = vectorizer.fit_transform(movies["content"])
    # sample = one_hot_matrix.toarray()[:4]
    sim = cosine_similarity(one_hot_matrix, one_hot_matrix)

    def recommend_movies(movie_index, cosine_sim_matrix, k=5):
        # 주어진 영화와 모든 영화 간의 유사도 점수를 가져옵니다.
        sim_scores = list(enumerate(cosine_sim_matrix[movie_index]))

        # 유사도에 따라 영화들을 정렬합니다.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 가장 유사한 k+1개의 영화를 선택합니다(자기 자신을 제외).
        sim_scores = sim_scores[1 : k + 1]

        # 추천된 영화의 인덱스를 가져옵니다.
        movie_indices = [i[0] for i in sim_scores]

        # 추천된 영화의 제목을 반환합니다.
        return movies["title"].iloc[movie_indices]

    recommended = recommend_movies(0, sim, k=10)
    print(recommended)


if __name__ == "__main__":
    main()
