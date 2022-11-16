import traceback

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time
from progress.bar import IncrementalBar
from itertools import combinations as comb


def Gnp(nodes, p=1, edges=0):
    g = nx.empty_graph(nodes, create_using=nx.Graph)

    if (p < 1):
        with IncrementalBar(max=nodes) as bar:
            for node1 in g.nodes():
                for node2 in g.nodes():
                    if not g.has_edge(node1, node2) and node1 != node2 and rd.random() < p:
                        try:
                            g.add_edge(node1, node2)
                        except:
                            print(traceback.format_exc())
                            continue
                bar.next()
            bar.finish()
    else:
        with IncrementalBar(max=edges) as bar:
            i = 0
            while i < edges:
                node1 = rd.randint(0, nodes)
                node2 = rd.randint(0, nodes)

                if not g.has_edge(node1, node2) and node1 != node2:
                    g.add_edge(node1, node2)
                    i += 1
                    bar.next()
            bar.finish()

    return g


def calculateGraph(G):
    degrees = []

    for node in G.nodes():
        degrees.append(nx.degree(G, node))

    print(f"Nº de nós: {G.number_of_nodes()}")
    print(f"Nº de links: {G.number_of_edges()}")
    print(f"Grau médio: {np.mean(degrees)}")
    print(f"Densidade: {nx.density(G)}")
    # print(f"Distância média: {nx.average_shortest_path_length(G)}")
    print(f"Cluster global: {nx.transitivity(G)}")
    print(f"Cluster médio: {nx.average_clustering(G)}")


def drawTheGraph(graph):
    fig = plt.figure(figsize=(10, 10))

    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    # Gcc = graph.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(graph, seed=10396953)
    nx.draw_networkx_nodes(graph, pos, ax=ax0, node_size=20, node_color="#5e0a1e")
    nx.draw_networkx_edges(graph, pos, ax=ax0, alpha=0.4, arrowstyle='->', arrowsize=5)
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


def saveThePlot(dicData, type, path=""):
    nodesAxis = [i for i in dicData["vertices"]]
    timeAxis = [i for i in dicData["mediasTempo"]]

    print(f"Tempos: {len(timeAxis)}; Vertices: {len(nodesAxis)}")

    plt.plot(timeAxis, nodesAxis, label=type)
    plt.xlabel("Médias de tempo em seg.")
    plt.ylabel("#Vértices")
    plt.scatter(timeAxis, nodesAxis, c='red')

    for i, text in enumerate(dicData["vertices"]):
        plt.annotate(text, (timeAxis[i], nodesAxis[i]))

    if path != "":
        plt.savefig(path)


n = [16, 64, 256, 1024, 4096, 16384]#, 65536]
dicData = {"vertices": n,
           "mediasTempo": [],
           "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []
    p = 0.1

    for i in range(10):
        t0 = time.time()

        graph = Gnp(value, p=p)
        # calculateGraph(graph)
        # drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1 - t0)
        print(f"Grafo {process}/10 gerado.")
        process += 1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)

saveThePlot(dicData, "Teste individual com P fixo")

dicData.clear()
dicData = {"vertices": n,
           "mediasTempo": [],
           "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []
    p = 10 / value

    for i in range(10):
        t0 = time.time()

        graph = Gnp(value, p=p)
        # calculateGraph(graph)
        # drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1 - t0)
        print(f"Grafo {process}/10 gerado.")
        process += 1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)

saveThePlot(dicData, "Teste individual com P variável")

dicData.clear()
dicData = {"vertices": n,
           "mediasTempo": [],
           "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []
    p = 0.1

    for i in range(10):
        t0 = time.time()

        M = np.random.binomial(value, p)
        graph = Gnp(value, edges=2 * M)
        # calculateGraph(graph)
        # drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1 - t0)
        print(f"Grafo {process}/10 gerado.")
        process += 1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)

saveThePlot(dicData, "Teste sorteando arestas com P fixo e n sendo #nós")

dicData.clear()
dicData = {"vertices": n,
           "mediasTempo": [],
           "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []
    p = 10 / value

    for i in range(10):
        t0 = time.time()

        M = np.random.binomial(value, p)
        graph = Gnp(value, edges=2 * M)
        # calculateGraph(graph)
        # drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1 - t0)
        print(f"Grafo {process}/10 gerado.")
        process += 1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)

saveThePlot(dicData, "Teste sorteando arestas com P variável e n sendo #nós")

dicData.clear()
dicData = {"vertices": n,
           "mediasTempo": [],
           "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []
    p = 10 / value

    for i in range(10):
        t0 = time.time()

        maxEdges = int(len([i for i in comb(range(value),2)])/2)
        M = np.random.binomial(maxEdges, p)
        graph = Gnp(value, edges=2 * M)
        # calculateGraph(graph)
        # drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1 - t0)
        print(f"Grafo {process}/10 gerado.")
        process += 1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)

saveThePlot(dicData, "Teste sorteando arestas com P variável e n sendo #máximo de arestas")

dicData.clear()
dicData = {"vertices": n,
           "mediasTempo": [],
           "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []
    p = 10 / value

    for i in range(10):
        t0 = time.time()

        maxEdges = int(len([i for i in comb(range(value), 2)]) / 2)
        M = np.random.binomial(maxEdges, p)
        graph = Gnp(value, edges=2 * M)
        # calculateGraph(graph)
        # drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1 - t0)
        print(f"Grafo {process}/10 gerado.")
        process += 1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)

saveThePlot(dicData, "Teste sorteando arestas com P variável e n sendo #máximo de arestas")

plt.legend(loc="lower right", shadow=True)
plt.savefig("D:\\Documentos\\UNIRIO\\5º Período\\Introdução à Ciência de Redes\\Plots\\PlotModelos")
plt.show()
