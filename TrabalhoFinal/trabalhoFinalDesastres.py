import os
from datetime import datetime

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from progress.bar import IncrementalBar

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Dado:
    def __init__(self, pontos, ):
        self.pontos = pontos
        self.dfSaida = None

    def retornaDataFrameVizinhos(self):
        pontos2 = pd.read_excel(self.pontos)

        # Criando cópia do dataframe para utilizar no loop
        temp = pontos2.copy()
        analise = pd.DataFrame(columns=['Dis No', 'Year', 'Neighbors'])

        with IncrementalBar('Gerando o Dataframe de viznhos', max=len(temp)) as bar:
            # Iterando as linhas dos dataframes
            for ix, r in pontos2.iterrows():
                for ixT, rT in temp.iterrows():
                    if r['Dis No'] == rT['Dis No']: #Ignorando caso os dados tenham o mesmo código identificador
                        continue

                    if r['Disaster Type'] != rT['Disaster Type']:  # ignorando caso os dados sejam de tipos de desastre diferentes
                        continue

                    date1 = datetime(int(r['Year']), int(r['Start Month']), 1)
                    date2 = datetime(int(rT['Year']), int(rT['Start Month']), 1)
                    if abs((date1.year - date2.year) * 12 + date1.month - date2.month) > 2: # Restrigindo a busca para uma diferença de 2 meses no máximo
                        continue

                    if rT['Country'] == r['Country']: # Ignorando caso os dados estejam em países diferentes
                        continue
                    analise = analise.append({'Dis No': r['Dis No'], 'Year': r['Year'], 'Neighbors': rT['Dis No']},
                                             ignore_index=True)
                bar.next()
            bar.finish()

        self.dfSaida.head()

        self.dfSaida.to_csv(os.getcwd() + "\Datasets\ListaVizinhos1995_2000.csv")

        return self.dfSaida

class Grafo:
    def __init__(self, dfDados):
        self.dfDados = dfDados
        self.G = None

    def loadGraph(self):
        self.G = nx.Graph()

        with IncrementalBar("Gerando grafo", max=self.dfDados.shape[0]) as bar:
            for index, row in self.dfDados.iterrows():
                self.G.add_edge(row['Dis No'], row['Neighbors'])
                bar.next()
            bar.finish()

    def calculateGraph(self):
        self.loadGraph()

        degrees = []

        for node in self.G.nodes():
            degrees.append(nx.degree(self.G, node))

        print(f"Nº de nós: {self.G.number_of_nodes()}")
        print(f"Nº de links: {self.G.number_of_edges()}")
        print(f"Grau médio: {np.mean(degrees)}")
        print(f"Densidade: {nx.density(self.G)}")
        # print(f"Distância média: {nx.average_shortest_path_length(G)}")
        print(f"Cluster global: {nx.transitivity(self.G)}")
        print(f"Cluster médio: {nx.average_clustering(self.G)}")

        self.drawTheGraph()

    def drawTheGraph(self):
        fig = plt.figure(figsize=(10, 10))

        degree_sequence = sorted((d for n, d in self.G.degree()), reverse=True)

        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        # Gcc = graph.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(self.G, seed=10396953)
        nx.draw_networkx_nodes(self.G, pos, ax=ax0, node_size=20, node_color="#5e0a1e")
        nx.draw_networkx_edges(self.G, pos, ax=ax0, alpha=0.4, arrowstyle='->', arrowsize=5)
        ax0.set_title("Connected components of G")
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")

        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    arqPontos = os.getcwd() + "\\NaturalDisasters.xlsx"
    print(arqPontos)

    dado = Dado(arqPontos)

    dfDados = dado.retornaDataFrameVizinhos()

    print(dfDados.head(100))

    dfDados.to_csv(os.getcwd() + "\\Datasets\\ListaVizinhos.csv")

    #dfDados = pd.read_csv(os.getcwd() + "\Datasets\ListaVizinhos300km_1995_2000.csv") #Utilizar esse caso passe um arquivo com os dados para o grafo

    grafo = Grafo(dfDados)

    grafo.calculateGraph()
