from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scipy.stats import entropy
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities,modularity


def apply(lst, *functions):
    """Apply functions to all elements
    of a list in a respective order and
    return it
    """
    lst_aux = lst
    for function in functions:
        lst_aux = list(map(function, lst_aux))
    return lst_aux


def remove_punctuation(text):
    '''Receives a text string and
    returns the list of word tokens on it
    '''
    tokenizer = RegexpTokenizer(r'\w\w+')
    return tokenizer.tokenize(text)


def lowercase(word_list):
    '''Receives a list of strings
    and return a list of all
    elements lower cased
    '''
    return list(map(lambda x: x.lower(), word_list))


def convert2lemma(word_list):
    '''Receives a word list and
    return the respective lemma list
    '''
    lemmatizer = WordNetLemmatizer()
    get_lemma = lambda word: lemmatizer.lemmatize(word)
    return list(map(get_lemma, word_list))

    
def remove_stpwrds(word_list, stop_list):
    '''Receives a word list and
    a stopwords list then return
    the word list without the stopwords
    '''
    return [i for i in word_list if i not in stop_list]


def graphfy(paragraphs):
    '''Receives word tokens lists then return a
    graph network created by it using the same
    rules described on the  InfraNodus paper:
    https://www.researchgate.net/publication/333067492_InfraNodus_Generating_Insight_Using_Text_Network_Analysis
    '''

    def add_edge(dict_edges, node1, node2, weight):
        '''Receive a dictionary of edges and return
        it with a new edge added that goes from
        node1 to node2 with the specified weight
        '''
        if node1 not in dict_edges.keys():
            dict_edges[node1] = {}
        if node2 not in dict_edges[node1].keys():
            dict_edges[node1][node2] = {'weight':0}
        dict_edges[node1][node2]['weight'] += weight
        return dict_edges

    def graphfy_paragraph(paragraph):
        '''Following the same rules of the InfraNodus
        paper, creates a network from a token word list
        '''
        dict_edges = {}
        length = len(paragraph)
        
        for index,word in enumerate(paragraph):  
            for ahead in range(1,min(4,length-index)):
                next_word = paragraph[index+ahead]
                dict_edges = add_edge(dict_edges,
                                      word,
                                      next_word,
                                      4-ahead)
        return dict_edges
    
    # Using the two functions described above,
    # we create a graph for each paragraph and
    # and combine them:
    
    dict_edges = {}
    for paragraph in paragraphs:
        p_edges = graphfy_paragraph(paragraph)
        for word in p_edges.keys():
            for next_word in p_edges[word].keys():
                
                weight = p_edges[word][next_word]['weight']
                dict_edges = add_edge(dict_edges,
                                      word,
                                      next_word,
                                      weight)

    
    # Create a Networkx graph object
    # from the list of edges
    
    graph = nx.DiGraph(dict_edges)

    # Setting attributes
    dict_name = {node:node for node in list(graph.nodes)}
    
    for attr in ('title','label'):
        nx.set_node_attributes(graph,
                               dict_name,
                               attr)
    
    return graph


def influential_nodes(graph,n_nodes=0):
    '''Receives a graph and use the
    Networkx's Betweenness Centrality 
    method to find the most influential
    nodes and returns the graph with
    each node centrality attribute, the
    centrality dictionary
    
    Optional Parameter
    ------------------
    n_nodes : amount of nodes to maintain
    in the graph, by removing less influential
    nodes. If equals to 0, the fucntion
    doesn't remove any node
    '''
    centrality_dict = nx.betweenness_centrality(graph, k=1000,
                                                weight='weight',
                                                endpoints=True)
    if n_nodes:
        node_list = list(graph.nodes)
        node_list.sort(key=lambda x: centrality_dict[x])
        remove_list = node_list[:-n_nodes]
        for node in remove_list:
            graph.remove_node(node)
    
    nx.set_node_attributes(graph,
                           centrality_dict, "size")
    return (graph, centrality_dict)


def nodes_communities(graph):
    '''Receives a graph and use the
    Networkx's Greedy Modularity Communities 
    method to find communities on the graph
    and returns it which each node with a new
    attribute of their respective communities
    '''
    communities = greedy_modularity_communities(nx.Graph(graph),
                                                weight='weight',
                                                best_n=5)
    community_dict = {}
    for node in graph.nodes:
        for i,community in enumerate(communities):
            if node in community:
                community_dict[node] = i+10
    
    nx.set_node_attributes(graph,
                           community_dict, "group")
    return graph,communities


def size_from_centrality(cen, max_cen):
    '''Receives a node's centrality
    and the max centrality in the graph
    then return a size to use in the graph's plot,
    with the sizes going from 1 to 100
    '''
    return max(1, min(100, (100/max_cen)*cen))
    
def top_cluster_percent(communities):
    '''Receives a list of node sets
    and return the calculated nodes
    percentage on the biggest set
    '''
    comm = apply(communities, len)
    
    return max(comm)/sum(comm)

def graph_entropy(graph):
    '''Receives a graph and return a
    float entropy of the top 4 nodes
    communities using the same method
    at the InfraNodus paper
    '''

    list_nodes = list(graph.nodes)
    
    get_cen = nx.get_node_attributes(graph, "size")
    list_nodes.sort(key=lambda node: get_cen[node])
    
    top4 = list_nodes[-4:]
    get_comm = nx.get_node_attributes(graph, "group")
    sets_lst = apply(top4, lambda node: get_comm[node])
    dict_prob = {}
    for i in sets_lst:
        if i in dict_prob.keys():
            dict_prob[i] += 0.25
        else:
            dict_prob[i] = 0.25
    return entropy(list(dict_prob.values()), base=2)
    
def classification(M,C,E):
    '''Receives a graph's communities
    modularity, the percentage of the
    top cluster and the entropy of the
    string of the top 4 nodes then
    returns the classification
    according to the Infranodus paper
    '''
    if M > 0.65 and C < 0.5 and E >= 1.5:
        return 'Dispersed'
    elif M <= 0.6 and M > 0.4 and C < 0.5 and E >= 1.5:
        return 'Diversified'
    elif M < 0.4 and M >= 0.2 and E > 0.5:
        return 'Focused'
    elif M > 0.4 and C >= 0.5 and E > 0.25 and E < 0.75:
        return 'Focused'
    elif M < 0.2:
        return 'Biased'
    else:
        return 'Undefined'