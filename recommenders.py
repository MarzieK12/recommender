"""
In this script we define functions for the recommender web
# application
# """

import pandas as pd
import numpy as np

# import random
# from utils import MOVIES, nmf_model, cos_sim_model


# def nmf_recommender(query, nmf_model, titles, k=10):
#     """This is an nmf-based recommender"""
#     return NotImplementedError

# def cos_sim_recommender(query, cos_sim_model, titles, k=10):
#     """This is an cosine-similarity-based recommender"""
#     return NotImplementedError


# def random_recommender(k=2):
#     if k > len(MOVIES):
#         print(f"Hey you exceed the allowed number of movies {len(MOVIES)}")
#         return []
#     else:
#         random.shuffle(MOVIES)
#         top_k = MOVIES[:k]
#         return top_k


# if __name__ == "__main__":
#     top2 = random_recommender()
#     print(top2)

def recommend_nmf(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    movie_title = pd.read_csv('movie_title.csv')

    # 1. construct new_user-item dataframe given the query
    df_new_user =  pd.DataFrame(query, columns=movie_title["title"], index=["new_user"]).fillna(0)

    # 2. scoring
    
    # calculate the score with the NMF model
    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix)
    P_new_user_matrix = model.transform(df_new_user)
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=movie_title["title"],
                         index = ['new_user'])

    # 3. ranking
    
    # filter out movies already seen by the user
    R_hat_new_user_filtered=R_hat_new_user.drop(query.keys(), axis=1)
    ranked = R_hat_new_user_filtered.T.sort_values(by=["new_user"], ascending=False).index.to_list()

    # return the top-k highest rated movie ids or titles
    recommendation = ranked[:4]
    return recommendation

def recommend_col(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """

    movie_title = pd.read_csv('movie_title.csv')
    R_df = pd.read_csv('R_df.csv', index_col=0)

    # 1. construct new_user-item dataframe given the query
    df_new_user =  pd.DataFrame(query, columns=movie_title["title"], index=["new_user"]).fillna(0)

    # 2. scoring
    similarity_scores, neighbor_ids = model.kneighbors(
    df_new_user,
    n_neighbors=15,
    return_distance=True
    )  
        
    neighbors_df = pd.DataFrame(
    data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )
    # 3. ranking
    neighborhood = R_df.iloc[neighbor_ids[0]]
    neighborhood_filtered = neighborhood.drop(query.keys(), axis=1)
    df_score = neighborhood_filtered.sum()
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()
    recommendations = df_score_ranked[:4]
    return recommendations #, df_score.sort_values(ascending=False)

