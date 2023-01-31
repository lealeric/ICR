import os
# import pandas as pd
# import networkx as nx
# import numpy as np
import tmdbsimple as tmdb
from dotenv import load_dotenv
# import matplotlib.pyplot as plt

# from datetime import datetime
# from progress.bar import IncrementalBar

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

tmdb.API_KEY = os.getenv('MOVIEDB_KEY')

search = tmdb.Search()
person1 = tmdb.People(search.person(query='Saoirse Ronan')['results'][0]['id'])
person2 = tmdb.People(search.person(query='Greta Gerwig')['results'][0]['id'])

person1Movies = person1.movie_credits()
person2Movies = person2.movie_credits()

worksTogether = {}

for movie in person1Movies['cast']:
    for movie2Cast in person2Movies['cast']:
        if movie['id'] == movie2Cast['id']:
            worksTogether[movie['id']] = [tmdb.Movies(movie['id']).info()['title'], movie['release_date']]

    for movie2Crew in person2Movies['crew']:
        if movie['id'] == movie2Crew['id'] and movie['id'] not in worksTogether.keys():
            worksTogether[movie['id']] = [tmdb.Movies(movie['id']).info()['title'], movie['release_date']]

for key in worksTogether.keys():
    print(worksTogether[key][0], worksTogether[key][1])




