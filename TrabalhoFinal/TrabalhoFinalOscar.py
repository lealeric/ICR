import requests
import os
import traceback
import winsound
import pandas as pd
import networkx as nx
import numpy as np
import tmdbsimple as tmdb
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from re import search
from dotenv import load_dotenv
from community import community_louvain
# from progress.bar import IncrementalBar

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

# # # INSTALAR kaleido E python-louvain PELO PIP # # #

load_dotenv()
tmdb.API_KEY = os.getenv('MOVIEDB_KEY')

class Dado:
    def __init__(self, data = None):
        self.dados = data
        self.worksTogether = {}
        self.notWorksTogether = {}
        self.metricas = None

         #pd.read_csv(self.caminhoArquivo, sep=';', encoding='utf-8')

    def getDados(self):
        return self.dados

    def getColuna(self, coluna):
        return self.dados[coluna]

    def getColunaPorIndice(self, indice):
        return self.dados.iloc[:, indice]

    def getFilmes(self, pessoa1 = None, pessoa2 = None, year = None):

        searchTMDB = tmdb.Search()

        if (pessoa1 in self.worksTogether.keys() and pessoa2 in self.worksTogether[pessoa1].keys() 
            or pessoa2 in self.worksTogether.keys() and pessoa1 in self.worksTogether[pessoa2].keys()
            or pessoa1 in self.notWorksTogether.keys() and pessoa2 in self.notWorksTogether[pessoa1]
            or pessoa2 in self.notWorksTogether.keys() and pessoa1 in self.notWorksTogether[pessoa2]):
            # print("Análise feita entre {} e {}".format(pessoa1, pessoa2))
            return       

        try:
            # person1 = tmdb.People(searchTMDB.person(query=pessoa1)['results'][0]['id'])
            # person2 = tmdb.People(searchTMDB.person(query=pessoa2)['results'][0]['id'])
            person1 = searchTMDB.person(query=pessoa1)['results'][0]['id']
            person2 = searchTMDB.person(query=pessoa2)['results'][0]['id']
        except Exception as e:
            print(e)
            print("Erro ao buscar pessoas")
            return

        if pessoa1 not in self.worksTogether.keys():
            self.worksTogether[pessoa1] = {}

        if pessoa1 not in self.notWorksTogether.keys():
            self.notWorksTogether[pessoa1] = []

        # Get requistions from a httprequest as json
        url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": tmdb.API_KEY,
            "sort_by": "release_date.asc",
            "with_people": "{},{}".format(person1, person2)
        }

        discover = requests.get(url, params=params).json()

        if not discover['results']:
            self.notWorksTogether[pessoa1].append(pessoa2)
            # print("Não há filmes em comum entre {} e {}".format(pessoa1, pessoa2))

        else:
            for movie in discover['results']:
                if pessoa2 not in self.worksTogether[pessoa1].keys():
                    self.worksTogether[pessoa1][pessoa2] = []

                try:
                    releaseYear = int(movie['release_date'][:4]) if movie['release_date'][:4] != "" else 0
                    if releaseYear not in self.worksTogether[pessoa1][pessoa2] and releaseYear != 0:
                        # movieTitle = tmdb.Movies(movie['id']).info()['title']
                        self.worksTogether[pessoa1][pessoa2].append(releaseYear)
                        # print("{} e {} trabalharam juntos em {} em {}".format(pessoa1, pessoa2, movieTitle, releaseYear))
                except KeyError as keyError:
                    print(keyError)
                    print("Erro ao buscar filme")
                    continue

        # person1Movies = person1.movie_credits()
        # person2Movies = person2.movie_credits()

        # for movie in (person1Movies['cast']):
        #     if pessoa1 not in self.worksTogether.keys():
        #         self.worksTogether[pessoa1] = {}
                
        #     if pessoa1 not in self.notWorksTogether.keys():
        #         self.notWorksTogether[pessoa1] = []
            

        #     for movie2Cast in person2Movies['cast']:
        #         if movie['id'] == movie2Cast['id']:
        #             if pessoa2 not in self.worksTogether[pessoa1].keys():
        #                 self.worksTogether[pessoa1][pessoa2] = []

        #             releaseYear = int(movie['release_date'][:4]) if movie['release_date'][:4] != "" else 0
        #             if releaseYear not in self.worksTogether[pessoa1][pessoa2] and releaseYear != 0:
        #                 movieTitle = tmdb.Movies(movie['id']).info()['title']
        #                 self.worksTogether[pessoa1][pessoa2].append(releaseYear)
        #                 # print("{} e {} trabalharam juntos em {} em {}".format(pessoa1, pessoa2, movieTitle, releaseYear))

        #         else:
        #             self.notWorksTogether[pessoa1].append(pessoa2)

        #     for movie2Crew in person2Movies['crew']:
                
        #         if movie['id'] == movie2Crew['id']:
        #             if pessoa2 not in self.worksTogether[pessoa1].keys():
        #                 self.worksTogether[pessoa1][pessoa2] = []

        #                 releaseYear = int(movie['release_date'][:4]) if movie['release_date'][:4] != "" else 0
        #                 if releaseYear not in self.worksTogether[pessoa1][pessoa2] and releaseYear != 0:
        #                     movieTitle = tmdb.Movies(movie['id']).info()['title']
        #                     self.worksTogether[pessoa1][pessoa2].append(releaseYear)
        #                     # print("{} e {} trabalharam juntos em {} em {}".format(pessoa1, pessoa2, movieTitle, releaseYear))
                    
        #         else:
        #             self.notWorksTogether[pessoa1].append(pessoa2)

        # for movie in (person1Movies['crew']):
            # if pessoa1 not in self.worksTogether.keys():
            #     self.worksTogether[pessoa1] = {}
                
            # if pessoa1 not in self.notWorksTogether.keys():
            #     self.notWorksTogether[pessoa1] = []
            
            # # movieTitle = tmdb.Movies(movie['id']).info()['title']

            # for movie2Cast in person2Movies['cast']:
            #     if movie['id'] == movie2Cast['id']:
            #         if pessoa2 not in self.worksTogether[pessoa1].keys():
            #             self.worksTogether[pessoa1][pessoa2] = []

            #         releaseYear = int(movie['release_date'][:4]) if movie['release_date'][:4] != "" else 0
            #         if releaseYear not in self.worksTogether[pessoa1][pessoa2] and releaseYear != 0:
            #             movieTitle = tmdb.Movies(movie['id']).info()['title']
            #             self.worksTogether[pessoa1][pessoa2].append(releaseYear)
            #             # print("{} e {} trabalharam juntos em {} em {}".format(pessoa1, pessoa2, movieTitle, releaseYear))

            #     else:
            #         self.notWorksTogether[pessoa1].append(pessoa2)

            # for movie2Crew in person2Movies['crew']:
                
            #     if movie['id'] == movie2Crew['id']:
            #         if pessoa2 not in self.worksTogether[pessoa1].keys():
            #             self.worksTogether[pessoa1][pessoa2] = []

            #             releaseYear = int(movie['release_date'][:4]) if movie['release_date'][:4] != "" else 0
            #             if releaseYear not in self.worksTogether[pessoa1][pessoa2] and releaseYear != 0:
            #                 movieTitle = tmdb.Movies(movie['id']).info()['title']
            #                 self.worksTogether[pessoa1][pessoa2].append(releaseYear)
            #                 # print("{} e {} trabalharam juntos em {} em {}".format(pessoa1, pessoa2, movieTitle, releaseYear))
                    
            #     else:
            #         self.notWorksTogether[pessoa1].append(pessoa2)

    def generateMetricas(self, dicMetricas):
        self.metricas = pd.DataFrame.from_dict(dicMetricas)

        self.metricas.to_csv(os.getcwd() + '\\TrabalhoFinal\\Grafos\\Métricas por ano.csv')
        
    def exportGraphToCSV(self, grafo, year):
        nx.write_edgelist(grafo, os.getcwd() + '\\TrabalhoFinal\\Grafos\\Grafo' + str(year) + '.csv', delimiter=',', data=False)
    
class Grafo:
    def __init__(self, dados = None, grafo = None):
        self.dados = dados
        self.dado = Dado(self.dados)
        self.metricasGrafo = {
            "ano" : [],
            "numeroDeNos" : [],
            "numeroDeArestas" : [],
            "grauMedio" : [],
            "densidade" : [],
            "clusterGlobal" : [],
            "clusterMedio" : [],
            "assortatividadeGeral" : [],
            "assortatividadeOscar" : [],
            "centralidadeGrau" : [],
            "centralidadeProximidade" : [],
            "centralidadeBetweeness" : []
        }

        if grafo is None:
            self.grafo = nx.Graph()
            # self.criarGrafo()
        else:
            self.grafo = grafo

    def createGraph(self):
        self.grafo.add_nodes_from(self.dados['name'])
        
        for index, row in self.dados.iterrows():
            for index2, row2 in self.dados.iterrows():
                if row['name'] == row2['name']:
                    continue

                if self.grafo.has_edge(row['name'], row2['name']):

                    return self.dados, self.grafo, self.dado
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

    def addNodes(self, datasetAno):
        for index, row in datasetAno.iterrows():
            if not self.grafo.has_node(row['name']):
                self.grafo.add_node(row['name'])                
                attrs = {row['name']: {"oscarWinner": row['winner'] == "True", "oscars": 0 if row['winner'] == "False" else 1, "year": row['year_film']}}
                nx.set_node_attributes(self.grafo, attrs)
            
            else:
                if row['winner'] == "True":
                    self.grafo.nodes[row['name']]['oscars'] += 1

        print("Nós criados com sucesso")
        print("O grafo possui {} nós".format((self.grafo.number_of_nodes())))

    def populateGraph(self, year): 
        for node in self.grafo.nodes():
            for node2 in self.grafo.nodes():
                if node == node2:
                    print("Mesma pessoa")
                    continue

                if self.grafo.has_edge(node, node2):
                    print("Aresta já existe")
                    continue

                try:
                    if node in self.dados.keys():
                        if node2 in self.dados[node].keys():
                            if year in self.dados[node][node2]:
                                self.grafo.add_edge(node, node2)
                                print("Aresta adicionada com sucesso")
                        
                    elif node2 in self.dados.keys():
                        if node in self.dados[node2].keys():
                            if year in self.dados[node2][node]:
                                self.grafo.add_edge(node, node2)
                                print("Aresta adicionada com sucesso")

                    else:
                        continue

                except Exception as e:
                    print(e)
                    print("Erro ao adicionar aresta entre {} e {}".format(node, node2))
                    traceback.print_exc()
                    continue

    def calculateGraph(self, year = None):
        degrees = []

        for node in self.grafo.nodes():
            degrees.append(nx.degree(self.grafo, node))

        print(f"Nº de nós: {self.grafo.number_of_nodes()}")
        print(f"Nº de links: {self.grafo.number_of_edges()}")
        print(f"Grau médio: {np.mean(degrees)}")
        print(f"Densidade: {nx.density(self.grafo)}")
        # print(f"Distância média: {nx.average_shortest_path_length(G)}")
        # print(f"Cluster global: {nx.transitivity(self.grafo)}")
        # print(f"Cluster médio: {nx.average_clustering(self.grafo)}")
        # print(f'Coeficiente de assortatividade geral: {nx.degree_assortativity_coefficient(self.grafo)}')
        # print(f'Coeficiente de assortatividade por vencedores do oscar: {nx.attribute_assortativity_coefficient(self.grafo, "oscarWinner")}')

        # print(f'Coeficiente de centralidade de grau: {nx.degree_centrality(self.grafo)}')
        # print(f'Coeficiente de centralidade de proximidade: {nx.closeness_centrality(self.grafo)}')
        # print(f'Coeficiente de centralidade de betweeness: {nx.betweenness_centrality(self.grafo)}')

        self.metricasGrafo["ano"].append(year)
        self.metricasGrafo["numeroDeNos"].append(self.grafo.number_of_nodes())
        self.metricasGrafo["numeroDeArestas"].append(self.grafo.number_of_edges())
        self.metricasGrafo["grauMedio"].append(np.mean(degrees))
        self.metricasGrafo["densidade"].append(nx.density(self.grafo))
        self.metricasGrafo["clusterGlobal"].append(nx.transitivity(self.grafo))
        self.metricasGrafo["clusterMedio"].append(nx.average_clustering(self.grafo))
        self.metricasGrafo["assortatividadeGeral"].append(nx.degree_assortativity_coefficient(self.grafo))
        self.metricasGrafo["assortatividadeOscar"].append(nx.attribute_assortativity_coefficient(self.grafo, "oscarWinner"))
        self.metricasGrafo["centralidadeGrau"].append(nx.degree_centrality(self.grafo))
        self.metricasGrafo["centralidadeProximidade"].append(nx.closeness_centrality(self.grafo))
        self.metricasGrafo["centralidadeBetweeness"].append(nx.betweenness_centrality(self.grafo))        

        # self.dado.generateMetricas(self.metricasGrafo)


    def drawTheGraph(self, year = None):
        fig = plt.figure(figsize=(24, 13.5))

        # for node in self.grafo.nodes():
        #     node['oscarWinner'] = self.dados.loc[self.dados['name'] == node]['winner'] == "True"


        # wins = dict.fromkeys(self.grafo.nodes(), 1)
        nodeShape = ['*' if nx.get_node_attributes(self.grafo, "oscarWinner")[v] else 'o' for v in self.grafo.nodes()]
        nodeSize = [nx.get_node_attributes(self.grafo, "oscars")[v] * 20 for v in self.grafo.nodes()]
        nodeColor = ["#ffd11a" if nx.get_node_attributes(self.grafo, "oscarWinner")[v] else "#e7eb10" for v in self.grafo.nodes()]

        degree_sequence = sorted((d for n, d in self.grafo.degree()), reverse=True)

        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        # Gcc = graph.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(self.grafo, seed=10396953)
        nx.draw_networkx_nodes(self.grafo, pos, ax=ax0, node_size=nodeSize, label="Nodes", node_shape=nodeShape , node_color=nodeColor)
        nx.draw_networkx_edges(self.grafo, pos, ax=ax0, alpha=0.4, arrowstyle='->', arrowsize=5)
        ax0.set_title("Grafo para o ano de " + str(year))
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, :])
        ax1.bar(*np.unique(degree_sequence, return_counts=True))
        ax1.set_title("Histograma de grau")
        ax1.set_xlabel("Grau")
        ax1.set_ylabel("# of Nós")

        fig.tight_layout()

        # plt.show()
        if not os.path.exists(os.getcwd() + '\\TrabalhoFinal\\Grafos'):
            os.makedirs(os.getcwd() + '\\TrabalhoFinal\\Grafos')
        fig.savefig(os.getcwd() + '\\TrabalhoFinal\\Grafos\\grafoOscar_' + str(year) + '.png')
            
    def drawTheGraphPlotly(self, year = None):
        edge_x = []
        edge_y = []
        pos = nx.spring_layout(self.grafo, seed=10396953)
        for edge in self.grafo.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in self.grafo.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='RdBu',
                reversescale=True,
                symbol = [17 if nx.get_node_attributes(self.grafo, "oscarWinner")[v] else 0  for v in self.grafo.nodes()],
                color=[],
                size= [10 + (10 * nx.get_node_attributes(self.grafo, "oscars")[v]) for v in self.grafo.nodes()],
                colorbar=dict(
                    thickness=15,   
                    title='Graus',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        node_text = []

        for node in self.grafo.nodes():
            node_text.append('{}'.format(node))

        for node, adjacencies in enumerate(self.grafo.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            # node_text.append('Conexões de {}: {}'.format(self.grafo.nodes()[node] , str(len(adjacencies[1]))))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Rede de colaborações entre os indicados ao Oscar no ano de ' + str(year),
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        if not os.path.exists(os.getcwd() + '\\TrabalhoFinal\\Grafos'):
            os.makedirs(os.getcwd() + '\\TrabalhoFinal\\Grafos')
        fig.write_image(os.getcwd() + '\\TrabalhoFinal\\Grafos\\grafoOscar_' + str(year) + '.png')
        fig.write_html(os.getcwd() + '\\TrabalhoFinal\\Grafos\\grafoOscar_' + str(year) + '.html')

    def community_layout(self, partition):
        """
        Compute the layout for a modular graph.


        Arguments:
        ----------
        g -- networkx.Graph or networkx.DiGraph instance
            graph to plot

        partition -- dict mapping int node -> int community
            graph partitions


        Returns:
        --------
        pos -- dict mapping int node -> (float x, float y)
            node positions

        """

        pos_communities = self._position_communities(partition, scale=3.)

        pos_nodes = self._position_nodes(partition, scale=1.)

        # combine positions
        pos = dict()
        for node in self.grafo.nodes():
            pos[node] = pos_communities[node] + pos_nodes[node]

        return pos

    def _position_communities(self, partition, **kwargs):

        # create a weighted graph, in which each node corresponds to a community,
        # and each edge weight to the number of edges between communities
        between_community_edges = self._find_between_community_edges(partition)

        communities = set(partition.values())
        hypergraph = nx.DiGraph()
        hypergraph.add_nodes_from(communities)
        for (ci, cj), edges in between_community_edges.items():
            hypergraph.add_edge(ci, cj, weight=len(edges))

        # find layout for communities
        pos_communities = nx.spring_layout(hypergraph, **kwargs)

        # set node positions to position of community
        pos = dict()
        for node, community in partition.items():
            pos[node] = pos_communities[community]

        return pos

    def _find_between_community_edges(self, partition):

        edges = dict()

        for (ni, nj) in self.grafo.edges():
            ci = partition[ni]
            cj = partition[nj]

            if ci != cj:
                try:
                    edges[(ci, cj)] += [(ni, nj)]
                except KeyError:
                    edges[(ci, cj)] = [(ni, nj)]

        return edges

    def _position_nodes(self, partition, **kwargs):
        """
        Positions nodes within communities.
        """

        communities = dict()
        for node, community in partition.items():
            try:
                communities[community] += [node]
            except KeyError:
                communities[community] = [node]

        pos = dict()
        for ci, nodes in communities.items():
            subgraph = self.grafo.subgraph(nodes)
            pos_subgraph = nx.spring_layout(subgraph, **kwargs)
            pos.update(pos_subgraph)

        return pos

if __name__ == '__main__':
    print("Iniciando")

    dfOscarBase = pd.read_csv(os.getcwd() + '\\TrabalhoFinal\\Datasets\\oscar_dataset\\the_oscar_award.csv', encoding='utf-8')
    for index, row in dfOscarBase.iterrows():
        if row['year_ceremony'] < 2020:
            dfOscarBase.drop(index, inplace=True)
            continue
            
        if search('ACTOR', row['category']) is not None or search('ACTRESS', row['category']) is not None or search('DIRECTING', row['category']) is not None: 
            continue
        else:
            dfOscarBase.drop(index, inplace=True)

    d = Dado(dfOscarBase)
    contador, contador2 = 1
    for index, row in dfOscarBase.iterrows():
        print("Analisando {} de {}".format(contador, dfOscarBase.shape[0]))
        for index2, row2 in dfOscarBase.iterrows():
            print("Analisando {} de {}".format(contador2, dfOscarBase.shape[0]))
            if index == index2 or row['name'] == row2['name']:
                continue
            try:
                d.getFilmes(row['name'], row2['name'])
            except Exception as e:
                print(e)
                continue
            contador2 += 1
        contador += 1

    # for key, value in d.worksTogether.items():
    #     print(key, value)

    currentYear = 2020
    yearchecked = False

    
    grafo = Grafo(dados=d.worksTogether)
    for index, row in dfOscarBase.iterrows():
        if row['year_ceremony'] != currentYear:
            print(row['year_ceremony'])    
            yearchecked = False
            currentYear = row['year_ceremony']

        if yearchecked:
            continue

        print(currentYear)
        df = dfOscarBase[dfOscarBase["year_ceremony"] == currentYear]
        grafo.addNodes(df)
        grafo.populateGraph(int(currentYear))
        grafo.calculateGraph(int(currentYear))
        grafo.drawTheGraphPlotly(int(currentYear))
        try:
            d.exportGraphToCSV(grafo.grafo, int(currentYear))
        except:
            traceback.print_exc()
            print("Erro ao gerar o csv do grafo")
        yearchecked = True

    print("Gerando o grafo com as comunidades")
    partition = community_louvain.best_partition(grafo.grafo)
    pos = grafo.community_layout(partition)
    nx.draw(grafo.grafo, pos, node_color=list(partition.values()))
    plt.savefig(os.getcwd() + '\\TrabalhoFinal\\Grafos\\grafoOscarComunidade.png')

    print("Gerando o csv com as métricas por ano")
    d.generateMetricas(grafo.metricasGrafo)


    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

    