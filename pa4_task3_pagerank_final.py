import numpy as np
from  scipy import sparse
from bs4 import BeautifulSoup
import time
import networkx as nx
import collections
import operator
import urlparse
import csv
import pickle
global urlbase
urlbase='http://cse.uta.edu/'

#Pagerank code from Samuel
###############Do not change below
def compute_PageRank(G, beta=0.85, epsilon=10**-4):
    '''
    Efficient computation of the PageRank values using a sparse adjacency 
    matrix and the iterative power method.
    
    Parameters
    ----------
    G : boolean adjacency matrix. np.bool8
        If the element j,i is True, means that there is a link from i to j.
    beta: 1-teleportation probability.
    epsilon: stop condition. Minimum allowed amount of change in the PageRanks
        between iterations.

    Returns
    -------
    output : tuple
        PageRank array normalized top one.
        Number of iterations.

    '''    
    #Test adjacency matrix is OK
    n,_ = G.shape
    assert(G.shape==(n,n))
    #Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T/beta #vector
    #Initialize
    ranks = np.ones((n,1))/n #vector
    time = 0
    flag = True
    while flag:        
        time +=1
        with np.errstate(divide='ignore'): # Ignore division by 0 on ranks/deg_out_beta
            new_ranks = G.dot((ranks/deg_out_beta)) #vector
        #Leaked PageRank
        new_ranks += (1-new_ranks.sum())/n
        #Stop condition
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False        
        ranks = new_ranks
    return(ranks, time)


################In this task, you will evaluate Pagerank on the UTA graph


def construct_digraph(file_name, output_file_name):
    #Your input file will have a structure similar to fileNamesToUUID.txt
    #   ie each line has two columns separated by | (i.e. filename, url)
    # Analyze the file in two passes:
    #   in the first pass, assign node ids to each url/file
    #       for eg, web page i on line i has node index i
    #       be careful though: it is possible that same url occurs multiple times in the document
    #       in that case treat them as two different pages
    #   in the second pass
    #       Create a directed graph as follows:
    #           each web page is a node
    #           parse the a tag in the document  and add an edge to where it points
    #           if the link is not in our corpus, ignore it
    #           if there a link from url X to url Y
    #           and url Y occurs multiple times in the file say in lines a,b,c
    #           and edge from X=>a, X=>b and X=>c
    #  DO NOT CONSTRUCT A DIRECTED GRAPH USING ADJACENCY MATRIX
    #   your computer will not have enough RAM
    # So instead store the data in the edge list format in output_file_name
    #   without explicitly constructing the graph
    # Write the following in output_file_name
    #   first line says the number of nodes in the graph
    #   for each edge (u,v) in the graph:
    #       write u,v in the output_file_name one line for each edge
    h= {}
    i = 0
    #k = 0
    g=nx.Graph()
    url_list = []
    nodes = 0
	#global n_edges
	
    with open(file_name) as myfile:
		for line in myfile:
			file = [item.strip() for item in line.split('|')]
			h[file[1]]= i
			#i = i+1
			#for x in range(len(file)):
			g.add_node(i)
			#print len(myfile)
  
			f = open(file[0])
			file_contents = f.read()
			#f.close()
			soup = BeautifulSoup(file_contents)
			for link in soup.findAll('a'):
				key= link.get('href')
				url_list.append(key)
				#print url_list
				
			for links in url_list:
					#links.urlsplit('#')
					#print links
					#urlparse.urldefrag(links)-
					
				url=urlparse.urljoin(urlbase,links)
				url.split('#')[0]
					#print links
					#print links
				if(h.has_key(url)):
						#print 'hello'
					g.add_edge(i, h[url])
					#print g.edges()
					#ed = str(str(i)+","+str(h[url]))
					#print ed
					#edge_list.append(ed)
			i = i+1
		global n_nodes
		n_nodes = g.number_of_nodes()
		#print n_nodes
		global n_edges
		n_edges = g.number_of_edges()
		#print n_edges
		print g.edges()
        
        #print edge_list
			
		#print g.nodes()
		#print "******"
		#print sorted(h.items(),key = operator.itemgetter(1))
    with open(output_file_name, 'w') as csvfile:
        edgewriter = csv.writer(csvfile, delimiter=',')
        for edge in g.edges():
            edgewriter.writerow(edge)
		#'''f = open(output_file_name, 'wb+')
		#outcsv = csv.writer(f)
		#for edge in g.edges():
		#	outcsv.write(edge)
		#	f.write("\n")
		#nx.write_edgelist(g,output_file_name,delimiter=',', data = False)
		#f.write(g.edges())
		#f.write(str(g.edges())+"\n")
		#f.close()'''
			#file_contents = f.read()
			#node_index=file_contents
			#print node_index
    #####################Task t3a: your code below#######################
    #####################Task t3a: your code below#######################
    pass


def construct_sparse_graph_dictionary_of_keys(graph_file_name):
    #If you create a graph for UTA using traditional methods
    #   such as adjacency matrix (which we will need for pagerank)
    #   it might take tens of GB
    # So we will represent the graph as a sparse matrix
    # In this code, you will read the input file (graph that you wrote in construct_digraph
    #   and convert it to a sparse matrix with dictionary of keys (DoK) encoding
    #   you might want to read https://scipy-lectures.github.io/advanced/scipy_sparse/storage_schemes.html
    #       or http://scipy-lectures.github.io/advanced/scipy_sparse/
    #####################Task t3b: your code below#######################
	G = None
    #####################Task t3b: your code below#######################
    
	with open(graph_file_name,'r') as f:
		print n_nodes
		G = sparse.dok_matrix((n_nodes,n_nodes),dtype=np.float64)
		for line in f.readlines():
			origin, destiny = (int(x) for x in line.split(','))
			G[origin,destiny]= 1.0 * (origin != destiny)
	
	#return(dokm)
	print (G.todense())
    #print(dokm)
    #print "Graph size through DoK is ", G.nbytes
	return G 

def construct_sparse_graph_compressed_sparse_row(graph_file_name):
    #If you create a graph for UTA using traditional methods
    #   such as adjacency matrix (which we will need for pagerank)
    #   it might take tens of GB
    # So we will represent the graph as a sparse matrix
    # In this code, you will read the input file (graph that you wrote in construct_digraph
    #   and convert it to a sparse matrix with compressed sparse row (CSR) format
    #   you might want to read https://scipy-lectures.github.io/advanced/scipy_sparse/storage_schemes.html
    #       or http://scipy-lectures.github.io/advanced/scipy_sparse/
    #####################Task t3c: your code below#######################
    G = None
    row = []
    col = []    
	
    with open(graph_file_name,'r') as f:
        for line in f.readlines():
            a, b = (int(x) for x in line.split(','))
            row.append(b)
            col.append(a)
    G=sparse.csr_matrix(([1.0]*n_edges,(row,col)),shape=(n_nodes,n_nodes))
    print (G.todense())
    return(G)
    #print csr
    #####################Task t3c: your code below#######################
    #print "Graph size through CSR is ", G.nbytes
    #return G 

def construct_sparse_graph_networkx(graph_file_name):
    #In this task, we will compare our method with NetworkX
    #   one of the state of the art graph analytics platorm
    # Networkx has a from_edgelist function that accepts an array of edges
    #   and construct a graph from it.
    #Read the input file and popular the variable edge_list
    #####################Task t3d: your code below#######################
    edge_list= []
    h=nx.Graph()
    #reader = csv.reader(graph_file_name)
    #for row in reader:
		#row[-1] = float(row[-1]) # convert capacity to float
        # add other attributes to the dict() below as needed...
        # e.g. you might add weights here as well.
		#edge_list=yield (row[0],row[1])
    with open(graph_file_name) as f:
		for lines in f:
			#h = nx.parse_edgelist(lines, nodetype = int)
			file = [item.strip() for item in lines.split(',')]
			ed1 = str("("+str(file[0])+","+str(file[1])+")")
			edge_list.append(file)
		#print edge_list
			#file = [item.strip() for item in line.split(',')]
	#	edge_list = lines
		#print edge_list
    #edge_list = [line.split(',') for line in open(graph_file_name)]		 
    #edge_list = open(graph_file_name).readlines()
    #print edge_list
    #dg = pickle.load(open(graph_file_name))
    #print dg.edges()
   # H=nx.Graph()
    #H.add_path(graph_file_name)
    #H=nx.path_graph(graph_file_name)
	#G.add_nodes_from(H)
    #edge_list = G.add_edges_from(H.edges())
    #print edge_list
    #####################Task t3d: your code below#######################
    G = nx.from_edgelist(edge_list, create_using=nx.DiGraph())
    #print (G.edges)
    return G


def persist_pagerank(graph_file_name, output_file_name):
    #####################Task t3e: your code below#######################
    #Construct the graph using some construct algorithm
    # Compute the pagerank
    #For each web page compute its pagerank (score) and write it in output_file_name
	G=construct_sparse_graph_dictionary_of_keys(graph_file_name)
	#b= nx.pagerank(G)
	#b = compute_PageRank(G, beta=0.85, epsilon=10**-4)
	b=compute_PageRank(G, beta=0.85, epsilon=10**-4)
	print(output_file_name)
	with open(output_file_name, 'w') as f:
		f.write(str(b))
		f.close()
    #####################Task t3e: your code below#######################
	pass 

def compare_pagerank_algorithms(graph_file_name):
    algo_name = ["PageRank-DOK", "PageRank-CSR", "PageRank-NetworkX"]
    algo_fns = [construct_sparse_graph_dictionary_of_keys, construct_sparse_graph_compressed_sparse_row, construct_sparse_graph_networkx]

    for i in range(len(algo_name)):
        print "Testing:", algo_name[i]

        start_time = time.time()
        G = algo_fns[i](graph_file_name)
        end_time = time.time()

        time_for_graph_construction = end_time - start_time

        start_time = time.time()
        if algo_name[i] == "PageRank-NetworkX":
            nx.pagerank(G)
        else:
            compute_PageRank(G)

        end_time = time.time()
        time_for_pagerank_computation = end_time - start_time
        total_time = time_for_graph_construction + time_for_pagerank_computation


        print "Time for graph, page rank and total", time_for_graph_construction, time_for_pagerank_computation, total_time

		
		
construct_digraph('fileNamesToUUID.txt',"graph.txt")
construct_sparse_graph_dictionary_of_keys("graph.txt")
construct_sparse_graph_compressed_sparse_row("graph.txt")
#construct_sparse_graph_networkx("graph.txt")
#persist_pagerank("graph.txt", "pagerank.txt")
#compare_pagerank_algorithms("graph.txt")