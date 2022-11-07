import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

pathData = "D:\\Documentos\\UNIRIO\\5º Período\\Introdução à Ciência de Redes\\Datasets\\protein.edgelist.txt"

def loadGraph(path):
    #G = nx.DiGraph() #Gera um grafo direcionado
    G = nx.Graph()  #Gera um grafo não direcionado

    with open(path, 'r') as file:
        for item in file:
            G.add_edge(int(str(item).split("	")[0]), int(str(item).split("	")[1]))

    return G

def calculateGraph(G):
    degrees=[]

    for node in G.nodes():
        degrees.append(nx.degree(G,node))

    print(f"Nº de nós: {G.number_of_nodes()}")
    print(f"Nº de links: {G.number_of_edges()}")
    print(f"Grau médio: {np.mean(degrees)}")
    print(f"Densidade: {nx.density(G)}")
    #print(f"Distância média: {nx.average_shortest_path_length(G)}")
    print(f"Cluster global: {nx.transitivity(G)}")
    print(f"Cluster médio: {nx.average_clustering(G)}")

def drawTheGraph(graph):
    fig = plt.figure(figsize=(10, 10))

    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

    axgrid = fig.add_gridspec(5,4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    #Gcc = graph.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
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

if __name__ == '__main__':
    GArquivo = loadGraph(pathData) #Carrega grafo do arquivo

    GGerado = nx.binomial_graph(GArquivo.number_of_nodes(),nx.density(GArquivo))

    print("------Grafo gerado por arquivo-------")
    calculateGraph(GArquivo)  #Calcula e imprime as métricas

    print("\n\n------Grafo gerado aleatoriamente-------")
    calculateGraph(GGerado)

    drawTheGraph(GArquivo)  #Desenha o grafo

