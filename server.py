import numpy as np
import pandas as pd
import matrix_factorization_utilities
from flask import Flask
from flask import jsonify
from pprint import pprint

# Load user ratings
raw_dataset_df = pd.read_csv('feedbacks.csv')

# Load movie titles
movies_df = pd.read_csv('events.csv', index_col='event_id')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='user_id',
                            columns='event_id',
                            aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)

# Find all predicted ratings by multiplying U and M matrices
predicted_ratings = np.matmul(U, M)

app = Flask(__name__, static_url_path='')

@app.route("/")
def index():
    return send_from_directory('static', "index.html")

@app.route("/users/<id>")
def hello(id):
    id = int(id)
    reviewed_movies_df = raw_dataset_df[raw_dataset_df['user_id'] == id]
    reviewed_movies_df = reviewed_movies_df.join(movies_df, on='event_id')
    user_ratings = predicted_ratings[id - 1]
    movies_df['rating'] = user_ratings

    already_reviewed = reviewed_movies_df['event_id']
    recommended_df = movies_df[movies_df.index.isin(already_reviewed) == False]
    recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)

    response = recommended_df[["Activity", "Time", "Location", 'rating']].to_json(orient='index')
    pprint(response)
    return jsonify(response)

app.run()
