import networkx as nx
import pulp
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict 

class network():

    #initialise variables of network
    def __init__(self,nodes, edges):

        """
        ###################################################
        Declaration du reseau g (nodes et edges)
        ###################################################
        """

        self.g = nx.to_directed(nx.fast_gnp_random_graph(nodes,edges/nodes,directed=True))
        
        self.node_pose = {}
        for i in self.g.nodes():
            self.node_pose[i] = (random.uniform(1.0, 10.0),random.uniform(1.0, 10.0))

        self.G = nx.DiGraph(self.g)
        for i,j in self.g.edges:
            if self.g.get_edge_data(j,i) == None:
                self.G.add_edge(j,i)

        self.g = self.G
        
        """
        ###################################################
        Declaration des variables qui caracterise les nodes
        ###################################################
        """
        self.color = {}
        

        self.dict_capa = {}
        

        self.dict_used = {}
        

        self.dict_ratio = {}
        

        self.dict_delay = {}
        
        """
        ###################################################
        ###################################################
        """


        self.list_keys = ['shortest_path','min_delay','min_banwidth_sum','min_banwidth_square_sum','min_score','min_square_score']
        self.dict_prob = {}

        self.opti_path = {}

        self.target_dict = defaultdict(dict)        

        

                


    #attribuer les valeurs aux parametres du network
    def intiate_values(self):
        ###################################################
        for i, j in self.g.edges:
            self.color[i, j] = self.color[j, i] = (0,0,0,0.5)

        nx.set_edge_attributes(self.g, self.color, 'color')

        ###################################################
        for i, j in self.g.edges:
            self.dict_capa[i, j] = self.dict_capa[j, i] = round(random.uniform(1.0, 20.0), 0)

        nx.set_edge_attributes(self.g, self.dict_capa, 'capacity')

        ###################################################
        for i, j in self.g.edges:
            self.dict_used[i, j] = self.dict_used[j, i] = min(round(random.uniform(0.0, 15.0), 0),self.dict_capa[i, j])
            
        nx.set_edge_attributes(self.g, self.dict_used, 'used')

        ###################################################
        for i, j in self.g.edges:
            self.dict_ratio[i, j] = self.dict_ratio[j, i] = self.dict_used[i, j]/self.dict_capa[i, j]

        nx.set_edge_attributes(self.g, self.dict_ratio, 'ratio')

        ###################################################
        for i, j in self.g.edges:
            self.dict_delay[i, j] = self.dict_delay[j, i] = round(random.uniform(1.0, 20.0), 2)

        nx.set_edge_attributes(self.g, self.dict_delay, 'delay')

        ###################################################
        big_d = {}
        small_d = {}

        for link in self.g.edges:
            small_d = {}
            small_d['Source'] = link[0]
            small_d['Ratio'] = self.g.edges[link]['ratio']
        
            big_d[str(link)] = small_d
        temp = pd.DataFrame.from_dict(big_d,orient='index')

        temp = temp.sort_values(by=['Source','Ratio'])

        for i in temp.Source.unique():
            cpt=0
            for j in temp[temp['Source'] == i].index:
                cpt+=1
                temp.at[str(j),'Score'] = cpt       
        
        
        for link in self.g.edges:
            self.g.edges[link]['score'] = temp.at[str(link),'Score'] 

        for link in self.g.edges:
            if self.g.edges[link]['score'] != temp.at[str(link),'Score'] :
                self.g.edges[link]['score'] = temp.at[str(link),'Score'] 

        
        ###################################################

        self.dict_prob = self.dict_prob.fromkeys(self.list_keys)

        self.opti_path = dict([(key, []) for key in self.list_keys])

        ###################################################
        ###################################################
        
        #init colors
    def init_colors(self):
        color = {}
        for i, j in self.g.edges:
            color[i, j] = color[j, i] = (0,0,0,0.5)

        nx.set_edge_attributes(self.g, color, 'color')

        #draw the network
    def draw_initial_graph(self):
        nx.draw(self.g, with_labels=True)
        plt.savefig("Initial_Graph")
        plt.close()
        # plt.show()

    def net(self):
        return self.g.edges

    # def