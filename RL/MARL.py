
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:08:00 2021

@author: Thomas Tranchet
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense,GlobalMaxPooling1D,GlobalMaxPooling2D, LeakyReLU, Reshape, Input, ConvLSTM2D ,Flatten,Conv2D,MaxPooling2D 
from tensorflow.keras.optimizers import Adam

plt.style.use('ggplot')

class MARL():

    def __init__(self, net, source, target):
        iter = 10
        iter_actuel = 0
        current_iter = 0
        
        var_dict = {}
        net = net
        g = net.g
        source = source
        target = target

        var_dict = {}

        dict_res = defaultdict(dict)

        node_pose = net.node_pose

        expirience_replay = np.zeros((50,1,1,25,25),np.int8)
        
        q_network = _build_compile_model(test = 0)
  
    def intialise_variable(self):
        for (i, j) in g.edges:
            var_dict[i, j] = 0

        node_pose = net.node_pose
        g = net.g

##########################################################################################
    
    def get_path_as_list_tuples(self,path):
        solve_var = [] 
        for i in path.split("_"):
            var = str(i).split('(')
            var = var[1].split(',')
            var[0] = int(var[0])
            var[1] = int(var[1].strip(')'))
            solve_var.append(tuple(var))
        return solve_var


    def draw_path(self,solve_var):
        for link in g.edges:
            if link in solve_var:
                g.edges[link[0],link[1]]['color'] = (1,0,0,1) 
    
        colors = nx.get_edge_attributes(g,'color').values()

        nx.draw(g, pos = node_pose, 
            edge_color=colors, 
            with_labels=True)
        ## Save chosen path ##
        #plt.show()
        #plt.savefig("imgs/Q_table/Q_table_iter_{}".format(iter))
        plt.close()
        net.init_colors()
    
    def plot_graph(self,test):
        x = []
        y = []
        for i in range(iter):
            x.append(dict_res[i]["Number of nodes"])
            y.append(i)
            
        p = plt.plot(y, x)
        
        plt.title('Average on 10 last iter : {}'.format(test))
        plt.show(p)
        plt.close()
        
        
##########################################################################################
        
    def get_action(self,solve_var,current_node,mur = False):
        state = get_state(solve_var)
        
        sub_list = get_possible_actions(current_node)
        if mur == True:
            #print("give up")
            return random.choice(sub_list)[1]
        else:
            if np.random.rand() <= get_epsilon(): #Epsilon
                return random.choice(sub_list)[1]
            else :
                q_values = q_network.predict(state)
                return np.argmax(q_values[0])
          
    def get_possible_actions(self,current_node):
        sub_list = []
        for (i,j) in g.edges:
            if i == current_node:
                sub_list.append((i,j))
        return sub_list
    
    def get_list_possibles_nodes(self,current_node):
        sub_list = get_possible_actions(current_node)
        nodes_list = []
        for (i,j) in sub_list:
            nodes_list.append(j)
        #print(nodes_list)
        return nodes_list
    
    def get_state(self, solve_var,add_dim = True):
        state = np.zeros((net.nodes,net.nodes), np.int8)
        for (i,j) in g.edges:
            if (i,j) in solve_var:
                state[i,j] = 1
            elif i == source:
                state[i,j] = 2
            elif j == target:
                state[i,j] = 2
            else:
                state[i,j] = 0

        if add_dim:
            state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=0)
        #print(state.shape)
        return state
    
    def scale(self,X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom 
    
    def get_epsilon(self):
        return (1.03 - iter_actuel / iter ) 
        
    def give_final_reward(self,solve_var,neg_reward):
        cpt=0
        
        for i in range(50) :
            #print("cpt = {}".format(cpt))
            
            state = expirience_replay[i]
            #print("i = {} state = {}".format(i,state))
            if not (np.all(state == 0)):
                state = np.expand_dims(state, axis=0)
                #print("state : {}".format(state))
                action_taken = solve_var[cpt][1]
                #print("action taken = {}".format(action_taken))
                
                target = q_network.predict(state)
                #print("before : {}".format(target[0].shape))
                if len(solve_var) != 50:
                    #print("target with action taken : {} before : {}".format(action_taken,target[0][action_taken]))
                    #print("before : {}".format(target[0]))
                    target[0][action_taken] = target[0][action_taken] + 50/(neg_reward - cpt)
                    #print("A : {}".format(target[0]))
                    #print("target with action taken : {} after : {}".format(action_taken,target[0][action_taken]))
                else:
                    target[0][action_taken] = target[0][action_taken] - 100
                q_network.fit(state, target, epochs=1, verbose=0)
                target = q_network.predict(state)
                #print("After : {}".format(target[0]))
                cpt = cpt + 1
                
        expirience_replay = np.zeros((50,1,1,25,25),np.int8)
        
        
    def remember(self,solve_var):
        state = get_state(solve_var)
        state = np.expand_dims(state, axis=0)
        #print("Rmember state shap : {}".format(state.shape))
        #print("Rmember experience replay shap : {}".format(expirience_replay[current_iter].shape))
        for i in range (50):
            result = np.all(state==expirience_replay[i])
            if not result:
                expirience_replay[current_iter] = state
                
        
    def _build_compile_model(self,test,solve_var=[]):
        state = get_state(solve_var,False)
        model = Sequential()
        
        if test==0:
            model.add(ConvLSTM2D (16, kernel_size=(2, 25), activation=LeakyReLU(),input_shape = (1,1,25,25),data_format='channels_first'))            
            model.add(GlobalMaxPooling2D())
            
        model.add(Dense(net.nodes, activation='linear'))
        print(model.summary())
        model.compile(loss='mse', optimizer= Adam(learning_rate=0.1))
        return model

##########################################################################################     
    def DRL_table(self,bdw):
        test_list = [0]
        for test in test_list:
            q_network = _build_compile_model(test)
            var_dict = var_dict
            summ = 0
            iter = -1
            while iter < (iter):
                iter = iter + 1 
                iter_actuel = iter + 1
                current_node = source
                path = ""
                cpt = 0
                cpt_murs = 0
                solve_var = []
                j = source
                
                flag_finished = False
                while flag_finished != True and cpt < 50:
                    current_iter = cpt
                    if cpt_murs > 50:
                        j = get_action(solve_var,current_node,True)
                    else:
                        j = get_action (solve_var,current_node)
                    i = current_node
                    
                    #condition
                    #if (bdw + g.edges[i,j]['used']  <= g.edges[i,j]['capacity']):
                    if ((i,j) in get_possible_actions(i)):
                        path += "{}_".format(str((i,j)))
                        solve_var.append((i,j))
                        var_dict[i,j] = 1
                        current_node = j
                        cpt_murs = 0
                        cpt = cpt + 1
                        if j == target:
                            flag_finished = True
                        remember(solve_var)
                    else:
                        cpt_murs = cpt_murs + 1
                path = path[:-1]
                
                # if cpt == 50:
                #     print("ECHEC")
                    
                if (len(path) != 0) :
                    solve_var = get_path_as_list_tuples(path)
        
                    ## Draw chosen path red ##
                    draw_path(solve_var)
                    dict_res[iter]['Number of nodes'] = len(solve_var)
                    
                    if iter_actuel > 90:
                        summ = summ + len(solve_var)
                    # if iter_actuel > 98:
                    #     print(path)
                    #summ = summ + len(solve_var)
                    #print(get_epsilon())
                    #print("solve_var len : {}".format(len(solve_var)))
                    
                    give_final_reward(solve_var,len(solve_var))
                    
                    dict_res[iter]['Sum of delay'] = sum([g.edges[i, j]['delay'] for i, j in solve_var])
                    dict_res[iter]['Ratio Sum'] = sum([g.edges[i, j]['ratio'] for i, j in solve_var])
                    dict_res[iter]['Squared Ratio Sum'] = sum([g.edges[i, j]['ratio']**2 for i, j in solve_var])
                    dict_res[iter]['Score Sum'] = sum([g.edges[i, j]['score'] for i, j in solve_var])
                    dict_res[iter]['Squared Score Sum'] = sum([g.edges[i, j]['score']**2 for i, j in solve_var])
                else:
                    #print("no found path for iter : {}".format(iter))
                    iter = iter - 1
                intialise_variable()
            #plot_graph(summ/10)
            #print("{} : {}".format(test,summ/10))
                
                
    