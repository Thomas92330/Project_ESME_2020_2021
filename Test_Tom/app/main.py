from fastapi import FastAPI 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse

import networkx as nx
import pulp 
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 

import codecs
import random
import io

matplotlib.use('Agg')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
node_pose = {}
db = []
path = []
##############################################################
# Read index.html file which which is the home page
@app.get('/', response_class=HTMLResponse)
async def index():
    # find file in static
    file = codecs.open("static/index.html", "r")
    # make the page appear as the response class is HTMLResponse
    return file.read()

#############################################################
# go to this after the user has chosen to create its graph
@app.get('/generate/{nodes_edges}', response_class=HTMLResponse)
def image_nertworkx(nodes:int = 30, edges:int  = 3):
    g = nx.to_directed(nx.fast_gnp_random_graph(nodes,edges/nodes,directed=True))

    for i in g.nodes():
        node_pose[i] = (random.uniform(1.0, 10.0),random.uniform(1.0, 10.0))
    
    nx.draw(g, with_labels=True)
    plt.savefig("images/original.jpg")
    plt.close()
    dict_capa = {}
    for i, j in g.edges:
        dict_capa[i, j] = dict_capa[j, i] = round(random.uniform(1.0, 20.0), 0)
    
    nx.set_edge_attributes(g, dict_capa, 'capacity')
    
    color = {}
    for i, j in g.edges:
        color[i, j] = color[j, i] = (0,0,0,0.5)
        
    nx.set_edge_attributes(g, color, 'color')

    dict_used = {}
    for i, j in g.edges:
        dict_used[i, j] = dict_used[j, i] = min(round(random.uniform(0.0, 15.0), 0),dict_capa[i, j])
    
    nx.set_edge_attributes(g, dict_used, 'used')
    
    dict_ratio = {}
    for i, j in g.edges:
        dict_ratio[i, j] = dict_ratio[j, i] = dict_used[i, j]/dict_capa[i, j]
    
    nx.set_edge_attributes(g, dict_ratio, 'ratio')
    
    dict_delay = {}
    for i, j in g.edges:
        dict_delay[i, j] = dict_delay[j, i] = round(random.uniform(1.0, 20.0), 2)
    
    nx.set_edge_attributes(g, dict_delay, 'delay')

    big_d = {}
    small_d = {}

    for link in g.edges:
        small_d = {}
        small_d['Source'] = link[0]
        small_d['Ratio'] = g.edges[link]['ratio']
    
        big_d[str(link)] = small_d
    temp = pd.DataFrame.from_dict(big_d,orient='index')

    temp = temp.sort_values(by=['Source','Ratio'])

    for i in temp.Source.unique():
        cpt=0
        for j in temp[temp['Source'] == i].index:
            cpt+=1
            temp.at[str(j),'Score'] = cpt
    for link in g.edges:
        g.edges[link]['score'] = temp.at[str(link),'Score'] 

    for link in g.edges:
        if g.edges[link]['score'] != temp.at[str(link),'Score'] :
            g.edges[link]['score'] = temp.at[str(link),'Score'] 

    for link in  g.edges:
        if g.edges[link]['score'] != temp.at[str(link),'Score'] :
                g.edges[link]['score'] = temp.at[str(link),'Score'] 

    if len(db) > 0 :
        db.pop(0)
    db.append(nx.to_dict_of_dicts(g))
    
    file = codecs.open("static/graph_view.html", "r")
    return file.read()


@app.get('/graph')
async def get_graph():
    return db

@app.get("/vector_image")
def image_endpoint():
    file_like = open("images/original.jpg", mode="rb")
    return StreamingResponse(file_like, media_type="image/jpg")

@app.get("/opti_image")
def image_endpoint():
    file_like = open("images/opti_image.jpg", mode="rb")
    return StreamingResponse(file_like, media_type="image/jpg")

@app.get('/generate_path/{source_target}',response_class=HTMLResponse)
async def opti_path(source:int = 0, target:int  = 10):
    g = nx.Graph(db[0])
    list_keys = ['shortest_path','min_delay','min_banwidth_sum','min_banwidth_square_sum','min_score','min_square_score']
    dict_prob = {}
    dict_prob = dict_prob.fromkeys(list_keys)

    opti_path = {}
    opti_path = dict([(key, []) for key in list_keys])

    target_dict = defaultdict(dict)

    # binary variable to state a link is chosen or not
    for keys,prob in dict_prob.items():
        prob = pulp.LpProblem("%s" % keys, pulp.LpMinimize)
        var_dict = {}
        for (i, j) in g.edges:
            x = pulp.LpVariable("%s_(%s_%s)" % (keys,i,j), cat=pulp.LpBinary)
            var_dict[i, j] = x
        bdw = 1
        
        # objective function
        if keys == "shortest_path":
            prob += pulp.lpSum(var_dict[i, j] for i, j in g.edges), "Sum Node Count"
        elif keys == "min_delay":
            prob += pulp.lpSum([g.edges[i,j]['delay'] * var_dict[i, j] for i, j in g.edges]), "Sum delay"
        elif keys == "min_banwidth_sum":
            prob += pulp.lpSum([g.edges[i,j]['ratio'] * var_dict[i, j] for i, j in g.edges]), "Sum bandwidth ratio"
        elif keys == "min_banwidth_square_sum":
            prob += pulp.lpSum([g.edges[i,j]['ratio'] ** 2 * var_dict[i, j] for i, j in g.edges]), "Sum square bandwidth ratio"
        elif keys == "min_score":
            prob += pulp.lpSum([g.edges[i,j]['score'] * var_dict[i, j] for i, j in g.edges]), "Sum score"
        elif keys == "min_square_score":
            prob += pulp.lpSum([(g.edges[i,j]['score'] ** 2 * var_dict[i, j]) for i, j in g.edges]), "Sum square score"
            
        # constraints
        for node in g.nodes:
            rhs = 0
            if node == source:
                rhs = -1
            elif node == target:
                rhs = 1
            prob += pulp.lpSum([var_dict[i, k] for i, k in g.edges if k == node]) - pulp.lpSum([var_dict[k, j] for k, j in g.edges if k == node]) == rhs
        
        # constraints on capacity
        for i,k in g.edges:
            prob += var_dict[i, k]*bdw + g.edges[i,k]['used']  <=g.edges[i,k]['capacity']

        # solve
        prob.solve()
        print("\n\n" + str(keys))
        print(pulp.LpStatus[prob.status])
        print(pulp.value(prob.objective))
        for link in g.edges:
            if var_dict[link].value() == 1.0:
                print(link, end=" , ")
                opti_path[keys].append(link)

    solve_var = []
    for i in prob.variables():
        if i.varValue == 1:
            var = str(i).split('(')
            var = var[1].split('_')
            var[0] = int(var[0])
            var[1] = int(var[1].strip(')'))
            solve_var.append(tuple(var))
    
    
    target_dict[keys]['Number of nodes'] = len(solve_var)
    target_dict[keys]['Sum of delay'] = sum([g.edges[i, j]['delay'] for i, j in solve_var])
    target_dict[keys]['Ratio Sum'] = sum([g.edges[i, j]['ratio'] for i, j in solve_var])
    target_dict[keys]['Squared Ratio Sum'] = sum([g.edges[i, j]['ratio']**2 for i, j in solve_var])
    target_dict[keys]['Score Sum'] = sum([g.edges[i, j]['score'] for i, j in solve_var])
    target_dict[keys]['Squared Score Sum'] = sum([g.edges[i, j]['score']**2 for i, j in solve_var])
    
    for link in g.edges:
        if var_dict[link].value() == 1.0:
            print(link, end=" ,")
            opti_path[keys].append(link)
            g.edges[link[0],link[1]]['color'] = (1,0,0,1) 

    colors = nx.get_edge_attributes(g,'color').values()

    nx.draw(g, pos = node_pose, 
        edge_color=colors, 
        with_labels=True)

    plt.savefig("images/{}.jpg".format(keys))
    plt.show()

    color = {}
    for i, j in g.edges:
        color[i, j] = color[j, i] = (0,0,0,0.5)

    nx.set_edge_attributes(g, color, 'color')

    if pulp.LpStatus[prob.status] != 'Infeasible'  :
        file = codecs.open("static/path_view.html", "r")
        return file.read()
    else: return ('Infeasible')