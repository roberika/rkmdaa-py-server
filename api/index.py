from flask import Flask, request
import pickle
import requests as rq
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

def load_model():
    if "anime_recommendation_model" not in globals():
        global anime_recommendation_model
        anime_recommendation_model = pickle.load(open('data/anime_recommendation_model.sav', 'rb'))

def load_user_list(username):
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
    return user_history, user_scores

def get_recommendation(anime_id):
    return anime_recommendation_model.loc[anime_recommendation_model.MAL_ID == int(anime_id)].values.flatten()[:-1]

def append_predicted_scores(anime_list, user_scores):
    return [[str(rec), "{:2.2f}".format(np.nan_to_num(user_scores.loc[user_scores.anime_id.isin(get_recommendation(rec))].rating.mean()))] for rec in anime_list]

@app.route('/recommend/<username>/<id>')
def get_recommendation_with_scores(username, id):
    user_history, user_scores = load_user_list(username)
    load_model()
    return append_predicted_scores(get_recommendation(id), user_scores)

@app.route('/recommend/<username>')
def get_recommendations_for_current_user_with_scores(username):
    user_history, user_scores = load_user_list(username)
    load_model()
    
    anime_ids = user_history.anime_id.to_list()
    recs = anime_recommendation_model.loc[anime_recommendation_model.MAL_ID.isin(anime_ids)].values.flatten()
    recs = [i for i in recs if i not in anime_ids]
    recs_sorted, indices = np.unique(np.array(recs), return_index=True)
    recs = [recs[index] for index in sorted(indices)]
    
    return append_predicted_scores(recs, user_scores)

@app.route('/history/<username>')
def get_user_history_with_scores(username):
    user_history, user_scores = load_user_list(username)
    return [[str(anime.anime_id), str(anime.rating)] for anime in user_history.values]
    
    

    
