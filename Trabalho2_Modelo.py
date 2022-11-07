import traceback

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time

def Gnp(n, p):
    g = nx.Graph()

    edges = []
    for i in range(n):
        elem = rd.randint(0,9999999)

        while elem in edges:
            elem = rd.random()

        edges.append(elem)
        g.add_node(elem)
        i += 1

    for i in range(n):
        for j in range(n):
            if edges[i] != edges[j] and rd.random() < p:
                try:
                    g.add_edge(edges[i], edges[j])
                except:
                    print(traceback.format_exc())
                    continue

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


n = [16, 64, 256, 1024, 4096, 16384, 65536]
p = 0.1
dicData = {"vertices": n,
        "mediasTempo": [],
        "mediasDensidade": []}

for index, value in enumerate(n):
    print(f"Geração de grafo com {value} vértices.")
    process = 1
    timesElapsed = []
    densities = []

    for i in range(10):
        t0 = time.time()

        graph = Gnp(value,p)
        #calculateGraph(graph)
        #drawTheGraph(graph)

        t1 = time.time()

        densities.append(nx.density(graph))
        timesElapsed.append(t1-t0)
        print(f"Grafo {process}/10 gerado.")
        process+=1

    meanDensities = np.mean(densities)
    meanTime = np.mean(timesElapsed)
    print(f"Densidade média de {meanDensities}.")
    print(f"Tempo médio de {meanTime} segundos.")

    dicData["mediasTempo"].append(meanTime)
    dicData["mediasDensidade"].append(meanDensities)


#drawTheGraph(graph)
#calculateGraph(graph)

logEdges = [np.log(i) for i in dicData["vertices"]]
logTempo= [np.log(i) for i in dicData["mediasTempo"]]

plt.plot(logTempo, logEdges)
plt.xlabel("Log das médias de tempo")
plt.ylabel("Log dos vértices")
plt.scatter(logTempo, logEdges, edgecolors='red')

for i, text in enumerate (dicData["vertices"]):
    plt.annotate(text,(logTempo[i], logEdges[i]))

plt.savefig("D:\\Documentos\\UNIRIO\\5º Período\\Introdução à Ciência de Redes\\Plots\\PlotModelo3")
plt.show()

