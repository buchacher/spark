from flask import Flask
from flask import render_template
from flask import request
from data_manipulation import movies_id
from data_manipulation import movies_title
from data_manipulation import movies_year
from data_manipulation import genre_movies
from data_manipulation import highest_rating
from data_manipulation import highest_views
from data_manipulation import user_history
from data_manipulation import movies_watched
from data_manipulation import genre_list
from data_manipulation import genre_list_multiple_users
import pandas

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/handle_movie_id_data', methods=['POST'])
def handle_movie_id_data():
    movieIDFormEntry = request.form['movieIDFormEntry']
    df = movies_id(movieIDFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Movie ID", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_title_data', methods=['POST'])
def handle_movie_title_data():
    movieTitleFormEntry = request.form['movieTitleFormEntry']
    df = movies_title(movieTitleFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Movie Title", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_year_data', methods=['POST'])
def handle_movie_year_data():
    movieYearFormEntry = request.form['movieYearFormEntry']
    df = movies_year(movieYearFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Movie Year", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_genre_data', methods=['POST'])
def handle_movie_genre_data():
    movieGenreFormEntry = request.form['movieGenreFormEntry']
    df = genre_movies(movieGenreFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Movie Genre", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_top_ratings_data', methods=['POST'])
def handle_movie_top_ratings_data():
    movieTopRatingsFormEntry = request.form['movieTopRatingsFormEntry']
    df = highest_rating(movieTopRatingsFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Top Rated Movies", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_top_views_data', methods=['POST'])
def handle_movie_top_views_data():
    movieTopViewsFormEntry = request.form['movieTopViewsFormEntry']
    df = highest_views(movieTopViewsFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Top Viewed Movies", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_user_history_data', methods=['POST'])
def handle_movie_user_history_data():
    movieUserHistoryFormEntry = request.form['movieUserHistoryFormEntry']
    df = user_history(movieUserHistoryFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="User Watch History", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_user_watched_data', methods=['POST'])
def handle_movie_user_watched_data():
    movieUserWatchedFormEntry = request.form['movieUserWatchedFormEntry']
    df = movies_watched(movieUserWatchedFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="User Watch History", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/handle_movie_user_favourite_genre_data', methods=['POST'])
def handle_movie_user_favourite_genre_data():
    movieUserFavouriteGenreFormEntry = request.form['movieUserFavouriteGenreFormEntry']
    df = genre_list(movieUserFavouriteGenreFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="User Favourite Genre", tables=[pandaDF.to_html(classes='data')],
                           titles=pandaDF.columns.values)


@app.route('/vis_handle_movie_user_favourite_genre_data', methods=['POST'])
def vis_handle_movie_user_favourite_genre_data():
    movieUserFavouriteGenreFormEntry = request.form['movieUserFavouriteGenreFormEntry']
    df = genre_list(movieUserFavouriteGenreFormEntry, True)
    pandaDF = df.toPandas()
    genreColumnList = pandaDF['genres'].tolist()
    sumOfRatingsColumnList = pandaDF['Sum of Ratings'].tolist()
    return render_template('vis.html', title='Favourite Genres for User ID', userID=movieUserFavouriteGenreFormEntry,
                           max=max(sumOfRatingsColumnList), labels=genreColumnList, values=sumOfRatingsColumnList)


@app.route('/handle_movie_multiple_user_favourite_genre_data', methods=['POST'])
def handle_movie_multiple_user_favourite_genre_data():
    movieMultipleUserFavouriteGenreFormEntry = request.form['movieMultipleUserFavouriteGenreFormEntry']
    df = genre_list_multiple_users(movieMultipleUserFavouriteGenreFormEntry, True)
    pandaDF = df.toPandas()
    return render_template('results.html', resultType="Multiple User Favourite Genre",
                           tables=[pandaDF.to_html(classes='data')], titles=pandaDF.columns.values)
