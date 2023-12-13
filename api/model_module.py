import pickle
import requests as rq
import numpy as np
import pandas as pd
import sklearn

# Muat model jarak KNN antar anime yang sudah di hitung sebelumnya
def load_model():
    global anime_recommendation_model
    anime_recommendation_model = pickle.load(open('dataset/anime_recommendation_model.sav', 'rb'))

# Muat list anime yang user sudah tonton beserta skornya
# user_history berisi semua anime, digunakan untuk rekomendasi
# user_scores berisi anime yang memiliki skor bukan 0 (belum dinilai),
# digunakan untuk prediksi skor
# Difilter supaya tidak memuat ID diatas 48492 karena dataset hanya
# memiliki anime sampai ID tersebut
def load_user_list(username):
    global user_history
    global user_scores
    item_per_page = 1000
    i = 1
    url = "https://api.myanimelist.net/v2/users/"+str(username)+"/animelist?fields=node,list_status&limit=" + str(item_per_page)
    headers = {'X-MAL-CLIENT-ID': 'fe954b2111fb0add5b5b25bfaa638979', 'Accept': '*/*'}
    response = rq.get(url, headers=headers)
    user = response.json()["data"]
    while (len(response.json()["paging"]) != 0 and len(response.json()["paging"]["next"]) != 0):
        response = rq.get(url + "&offset=" + str(item_per_page * i), headers=headers)
        user = user | response.json()["data"]
        i += 1
    ids = [anime["node"]["id"] for anime in user]
    scores = [anime["list_status"]["score"] for anime in user]
    list_anime = {"anime_id": ids, "rating": scores}
    user_history = pd.DataFrame({"anime_id": ids, "rating": scores})
    user_history = user_history.loc[user_history.anime_id <= 48492]
    user_scores = user_history.loc[user_history.rating != 0]

def get_recommendation(anime_id):
    return anime_recommendation_model.loc[anime_recommendation_model.MAL_ID == anime_id].values.flatten()[:-1]

def get_recommendations(anime_ids):
    recs = anime_recommendation_model.loc[anime_recommendation_model.MAL_ID.isin(anime_ids)].values.flatten()
    recs = [i for i in recs if i not in anime_ids]
    recs_sorted, indices = np.unique(np.array(recs), return_index=True)
    return [recs[index] for index in sorted(indices)]

def append_predicted_scores(anime_list):
    return [[rec, user_scores.loc[user_scores.anime_id.isin(get_recommendation(rec))].rating.mean()] for rec in anime_list]

def get_recommendation_with_scores(anime_id):
    return append_predicted_scores(get_recommendation(anime_id))

def get_recommendations_with_scores(anime_ids):
    return append_predicted_scores(get_recommendations(anime_ids))

def get_recommendations_for_current_user():
    return get_recommendations(user_history.anime_id.to_list())

def get_recommendations_for_current_user_with_scores():
    return get_recommendations_with_scores(user_history.anime_id.to_list())


    
