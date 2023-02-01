import os
import pandas as pd
import networkx as nx
import numpy as np
import tmdbsimple as tmdb
import matplotlib.pyplot as plt

from re import search
from dotenv import load_dotenv
from datetime import datetime
from progress.bar import IncrementalBar
from typing import List, Dict, Tuple, Union

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()
tmdb.API_KEY = os.getenv('MOVIEDB_KEY')

class Dado:
    def __init__(self, data = None):
        self.dados = data

         #pd.read_csv(self.caminhoArquivo, sep=';', encoding='utf-8')

    def getDados(self):
        return self.dados

    def getColuna(self, coluna):
        return self.dados[coluna]

    def getColunaPorIndice(self, indice):
        return self.dados.iloc[:, indice]

    def getFilmes(self, pessoa1 = None, pessoa2 = None, year = None):

        searchTMDB = tmdb.Search()
        person1 = tmdb.People(searchTMDB.person(query=pessoa1)['results'][0]['id'])
        person2 = tmdb.People(searchTMDB.person(query=pessoa2)['results'][0]['id'])

        person1Movies = person1.movie_credits()
        person2Movies = person2.movie_credits()

        worksTogether = {}

        for movie in (person1Movies['cast'] + person1Movies['crew']) :
            if movie['release_date'][:4] != str(year):
                continue
            movieTitle = tmdb.Movies(movie['id']).info()['title']
            for movie2Cast in person2Movies['cast']:
                if movie2Cast['release_date'][:4] != str(year):
                    continue

                if movie['id'] == movie2Cast['id'] and movie['id'] not in worksTogether.keys():
                    worksTogether[movie['id']] = [movieTitle, movie['release_date'][:4]]

            for movie2Crew in person2Movies['crew']:
                if movie2Crew['release_date'][:4] != str(year):
                    continue
                
                if movie['id'] == movie2Crew['id'] and movie['id'] not in worksTogether.keys():
                    worksTogether[movie['id']] = [movieTitle, movie['release_date'][:4]]

        # for movie in person1Movies['crew']:
        #     if movie['release_date'][:4] != str(year):
        #         continue
        #     movieTitle = tmdb.Movies(movie['id']).info()['title']
        #     for movie2Cast in person2Movies['cast']:
        #         if movie2Cast['release_date'][:4] != str(year):
        #             continue
        #         if movie['id'] == movie2Cast['id'] and movie['id'] not in worksTogether.keys():
        #             worksTogether[movie['id']] = [movieTitle, movie['release_date'][:4]]

        #     for movie2Crew in person2Movies['crew']:
        #         if movie2Crew['release_date'][:4] != str(year):
        #             continue
        #         if movie['id'] == movie2Crew['id'] and movie['id'] not in worksTogether.keys():
        #             worksTogether[movie['id']] = [movieTitle, movie['release_date'][:4]]

        return worksTogether
class Grafo:
    def __init__(self, dados = None, grafo = None):
        self.dados = dados
        self.grafo = grafo
        self.dado = Dado(self.dados)

        if grafo is None:
            self.grafo = nx.Graph()
            # self.criarGrafo()

    def criarGrafo(self):
        self.grafo.add_nodes_from(self.dados['name'])
        
        for index, row in self.dados.iterrows():
            for index2, row2 in self.dados.iterrows():
                if row['name'] == row2['name']:
                    continue

                if self.grafo.has_edge(row['name'], row2['name']):
                    continue

                print("Adicionando aresta entre {} e {}".format(row['name'], row2['name']))
                try:
                    if len(self.dado.getFilmes(row['name'], row2['name'], row['year_film'])) > 0:
                        self.grafo.add_edge(row['name'], row2['name'])
                        print("Adicionado com sucesso")
                except Exception as e:
                    print(e)
                    print("Erro ao adicionar aresta entre {} e {}".format(row['name'], row2['name']))
                    continue
        
        self.drawTheGraph()

    def criaNos(self, datasetAno):
        for index, row in datasetAno.iterrows():
            if not self.grafo.has_node(row['name']):
                self.grafo.add_node(row['name'])
                attrs = {row['name']: {"oscarWinner": row['winner']}}
                nx.set_node_attributes(self.grafo, attrs)

        print("Nós criados com sucesso")
        print("O grafo possui {} nós".format((self.grafo.number_of_nodes())))

    def populaGrafo(self, year): 
        for node in self.grafo.nodes():
            for node2 in self.grafo.nodes():
                if node == node2:
                    continue

                if self.grafo.has_edge(node, node2):
                    continue

                # print("Procurando filmes em comum entre {} e {} no ano de {}".format(node, node2, year))
                try:
                    if len(self.dado.getFilmes(node, node2, year)) > 0:
                        self.grafo.add_edge(node, node2)
                        # print("Adicionado com sucesso")
                except Exception as e:
                    print(e)
                    print("Erro ao adicionar aresta entre {} e {}".format(node, node2))
                    continue

    def calculateGraph(self):
        degrees = []

        for node in self.grafo.nodes():
            degrees.append(nx.degree(self.grafo, node))

        print(f"Nº de nós: {self.grafo.number_of_nodes()}")
        print(f"Nº de links: {self.grafo.number_of_edges()}")
        print(f"Grau médio: {np.mean(degrees)}")
        print(f"Densidade: {nx.density(self.grafo)}")
        # print(f"Distância média: {nx.average_shortest_path_length(G)}")
        print(f"Cluster global: {nx.transitivity(self.grafo)}")
        print(f"Cluster médio: {nx.average_clustering(self.grafo)}")

    def drawTheGraph(self):
        fig = plt.figure(figsize=(10, 10))

        degree_sequence = sorted((d for n, d in self.grafo.degree()), reverse=True)

        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        # Gcc = graph.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(self.grafo, seed=10396953)
        nx.draw_networkx_nodes(self.grafo, pos, ax=ax0, node_size=20, node_color="#5e0a1e", label="Nodes")
        nx.draw_networkx_edges(self.grafo, pos, ax=ax0, alpha=0.4, arrowstyle='->', arrowsize=5)
        ax0.set_title("Connected components of G")
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, :])
        ax1.bar(*np.unique(degree_sequence, return_counts=True))
        ax1.set_title("Degree histogram")
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("# of Nodes")

        fig.tight_layout()

        plt.show()
            


if __name__ == '__main__':
    print("Iniciando")

    data = pd.read_csv(os.getcwd() + '\\TrabalhoFinal\\Datasets\\oscar_dataset\\the_oscar_award.csv', encoding='utf-8')
    print(data.shape[0])

    grafo = Grafo()

    for index, row in data.iterrows():
        if row['year_ceremony'] < 2021:
            data.drop(index, inplace=True)
            continue
            
        if search('ACTOR', row['category']) is not None or search('ACTRESS', row['category']) is not None or search('DIRECTING', row['category']) is not None: 
            print(row['name'])
        else:
            data.drop(index, inplace=True)

    firstYear = 2020
    yearchecked = False

    for index, row in data.iterrows():
        if row['year_film'] != firstYear:
            print(row['year_film'])
            
            grafo.drawTheGraph()
            yearchecked = False
            firstYear = row['year_film']

        if yearchecked:
            continue

        print(firstYear)
        df = data[data["year_film"] == firstYear]
        grafo.criaNos(df)
        grafo.populaGrafo(firstYear) 
        grafo.calculateGraph()
        yearchecked = True


    grafo.drawTheGraph()
    

    