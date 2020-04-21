# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import copy

def generate_blackwhite_trajectory(bw_static_graph_tr, steps, step_size):
  mybase_graph_tr = utils_tf.data_dicts_to_graphs_tuple(bw_static_graph_tr)
  # List of lists: it contains the status of 200 graphs over 50 steps 
  ListofGraphSteps=[]

  c=0
  for static_graph in bw_static_graph_tr:
    graph = static_graph
    graph_steps=[]
    graph_steps.append(graph)
    for i in range(1,51):
      last_graph=graph_steps[-1].copy()

      BN_WN, WNList = getBlackNodesWithWhiteNeighboursList(last_graph)
      nodes=copy.deepcopy(last_graph["nodes"])
      if BN_WN: 
        for index in BN_WN:
          # If remaining time for propagation (RT) is greater than zero, decrement it and append last_graph to graph_steps
          if nodes[index][1]>0:
            nodes[index][1]-=1
            last_graph["nodes"]=copy.deepcopy(nodes)
            graph_steps.append(last_graph)
          # Set neighbours colors to black, set their RT to NumberOfWhiteNeighbours
          else:
            NumberOfWhiteNeighbours=len(WNList)
            for wn in WNList:
              nodes[wn][0]=1
              nodes[wn][1]=NumberOfWhiteNeighbours
            last_graph["nodes"]=copy.deepcopy(nodes)
            graph_steps.append(last_graph)
      else:
        graph_steps.append(last_graph)
    c=c+1
    ListofGraphSteps.append(graph_steps)
    

  k=0
  ListofGraphStepsCounter=0
  # Create a dictionary of i'th step and convert it to GraphTuple
  for tmpgs in ListofGraphSteps:
      tmpfilldictionary=[]
      tmpfilldictionary.append(tmpgs[k])
      print("ListofGraphStepsCounter: "+str(ListofGraphStepsCounter))
      ListofGraphStepsCounter=ListofGraphStepsCounter+1
      print("tmpgs["+str(k)+"]")
      print(tmpgs[k])

  tmpgraph_steps_tuple=utils_tf.data_dicts_to_graphs_tuple(tmpfilldictionary)
  # print("tmpgraph_steps_tuple")
  # print(tmpgraph_steps_tuple)
  tmpnodes_per_step = tf.TensorArray(
        dtype=tmpgraph_steps_tuple.nodes.dtype, size=51, element_shape=tmpgraph_steps_tuple.nodes.shape)

  gscount=0
  for i in range(51):
    for gs in ListofGraphSteps:
      filldictionary=[]
      filldictionary.append(gs[i])
      print("gscount")
      print(gscount)
      gscount=gscount+1
    gscount=0
    print("filldictionary")
    print(len(filldictionary))
    print("utils_tf.data_dicts_to_graphs_tuple(filldictionary)")
    print(utils_tf.data_dicts_to_graphs_tuple(filldictionary))
    tmpnodes_per_step = tmpnodes_per_step.write(i, utils_tf.data_dicts_to_graphs_tuple(filldictionary).nodes)
  print("Nodes_per_step writing is complete!")
  print("tmpnodes_per_step.stack()")
  print(tmpnodes_per_step.stack())

  # _, n = roll_out_physics(simulator, graph, steps, step_size)
  return mybase_graph_tr, tmpnodes_per_step.stack()


def mybase_graph(n):
  """Define a basic black-white system graph structure.

  Args:
    n: index of the node to be black at creation

  Returns:
    data_dict: dictionary with globals, nodes, edges, receivers and senders
        to represent a structure like the one above.
  """
  # Node features for graph 1. b/w, RT
  # b/w: black/white, black=1, white=0
  # RT: Remaining time before propagation (in steps)

  globals_1 = [1., 2., 3.]
  
  nodes_1 = [[1,1],  # Node 0
           [0,0],  # Node 1
           [0,0],  # Node 2
           [0,0],  # Node 3
           [0,0]]  # Node 4
  
  for index, value in enumerate(nodes_1):
      if index == n:
        nodes_1[index][0] = 1
        nodes_1[index][1] = 1
  
  # Edge features for graph 0.
  edges_1 = [[100., 200.],  # Edge 0
            [101., 201.],  # Edge 1
            [102., 202.],  # Edge 2
            [103., 203.],  # Edge 3
            [104., 204.],  # Edge 4
            [105., 205.]]  # Edge 5

  # The sender and receiver nodes associated with each edge for graph 0.
  senders_1 = [0,  # Index of the sender node for edge 0
              1,  # Index of the sender node for edge 1
              1,  # Index of the sender node for edge 2
              2,  # Index of the sender node for edge 3
              2]  # Index of the sender node for edge 4
  receivers_1 = [1,  # Index of the receiver node for edge 0
                2,  # Index of the receiver node for edge 1
                3,  # Index of the receiver node for edge 2
                0,  # Index of the receiver node for edge 3
                4]  # Index of the receiver node for edge 4
  
  data_dict_1 = {
    "globals": globals_1,
    "nodes": nodes_1,
    "edges": edges_1,
    "senders": senders_1,
    "receivers": receivers_1
  }
  return data_dict_1



def AnyBlackNode(nodes):
  black_node=False
  for n in nodes:
    if n[0]==1:
      black_node=True
      break
  return black_node

def getBlackNodesIndeces(nodes):
  i=0
  black_nodes_indeces=[]
  for n in nodes:
    if n[0]==1:
      black_nodes_indeces.append(i)
    i+=1
  return black_nodes_indeces

def getSendersIndeces(bn, senders):
  i=0
  indeces=[]
  for s in senders:
    if s==bn:
      indeces.append(i)
    i+=1
  return indeces
  
def getRespectiveRecievers(senderIndeces, recievers):
  indeces=[]
  for s in senderIndeces:
    indeces.append(recievers[s])
  return indeces

def getBlackNodesWithWhiteNeighboursList(graph):
  """
  Args:
    graph: graph as a python dictionary 

  Returns:
    BN_WN: List of Black nodes which have white neighbour nodes
    WNList: List of white neighbour nodes
  """
  nodes=graph["nodes"]
  senders=graph["senders"]
  recievers=graph["receivers"]

  BN_WN=[]
  WNList=[]
  # Get Black nodes indeces
  BNIndeces=getBlackNodesIndeces(nodes)
  # For each BN, Get senders indeces where these BN exist
  for BN in BNIndeces:
    senderIndecesContainsBN=getSendersIndeces(BN, senders)
    # Get -respectively- indeces of recievers (Neighbours set)
    respectiveRecievers=getRespectiveRecievers(senderIndecesContainsBN, recievers)
    # Check if those neighbours are whites
    for RR in respectiveRecievers:
      if nodes[RR][0]==0:
        if BN not in BN_WN:
          BN_WN.append(BN)
        WNList.append(RR)
  return BN_WN, WNList

# Needed declarations
SEED = 1
tf.reset_default_graph()
rand = np.random.RandomState(SEED)



# Base graphs for training.
possible_black_node_index_tr = (0, 5)
bw_batch_size_tr=200
first_black_nodes_tr = rand.randint(*possible_black_node_index_tr, size=bw_batch_size_tr)
#print(first_black_nodes_tr)


## A python list of data_dict graphs
bw_static_graph_tr = []
for i in first_black_nodes_tr:
  bw_static_graph_tr.append(mybase_graph(i))


# List of lists: it contains the status of 200 graphs over 50 steps 
ListofGraphSteps=[]


for static_graph in bw_static_graph_tr:
  graph = static_graph
  graph_steps=[]
  graph_steps.append(graph)
  for i in range(1,51):
    last_graph=graph_steps[-1].copy()

    BN_WN, WNList = getBlackNodesWithWhiteNeighboursList(last_graph)
    nodes=copy.deepcopy(last_graph["nodes"])
    if BN_WN: 
      for index in BN_WN:
        # If remaining time for propagation (RT) is greater than zero, decrement it and append last_graph to graph_steps
        if nodes[index][1]>0:
          nodes[index][1]-=1
          last_graph["nodes"]=copy.deepcopy(nodes)
          graph_steps.append(last_graph)
        # Set neighbours colors to black, set their RT to NumberOfWhiteNeighbours
        else:
          NumberOfWhiteNeighbours=len(WNList)
          for wn in WNList:
            nodes[wn][0]=1
            nodes[wn][1]=NumberOfWhiteNeighbours
          last_graph["nodes"]=copy.deepcopy(nodes)
          graph_steps.append(last_graph)
    else:
      graph_steps.append(last_graph)
  ListofGraphSteps.append(graph_steps)


print("bw_static_graph_tr length after sim: "+str(len(bw_static_graph_tr)))
print("ListofGraphSteps length after sim: "+str(len(ListofGraphSteps)))
print("ListofGraphSteps[0] length after sim: "+str(len(ListofGraphSteps[0])))
print("ListofGraphSteps[10] length after sim: "+str(len(ListofGraphSteps[10])))
print("ListofGraphSteps[100] length after sim: "+str(len(ListofGraphSteps[100])))

print("Step 10 of ListofGraphSteps[100] after sim: "+str(ListofGraphSteps[100][10]))

##### Below, I just want to convert ListofGraphSteps (list of lists) to nodes_per_step (TensorArray of GraphTuple types) #####


# k=0
# for tmpgs in ListofGraphSteps:
#     # Create a dictionary of i'th step and convert it to GraphTuple
#     tmpfilldictionary=[]
#     tmpfilldictionary.append(tmpgs[k])
#
# tmpgraph_steps_tuple=utils_tf.data_dicts_to_graphs_tuple(tmpfilldictionary)
# tmpnodes_per_step = tf.TensorArray(
#        dtype=tmpgraph_steps_tuple.nodes.dtype, size=51, element_shape=tmpgraph_steps_tuple.nodes.shape)
#
# gscount=0
# for i in range(51):
#   filldictionary=[]
#   for gs in ListofGraphSteps:
#     filldictionary.append(gs[i])
#     print("gscount")
#     print(gscount)
#     gscount=gscount+1
#   gscount=0
#   print("filldictionary")
#   print(len(filldictionary))
#   tmpnodes_per_step = tmpnodes_per_step.write(i, utils_tf.data_dicts_to_graphs_tuple(filldictionary).nodes)
# print("Nodes_per_step writing is complete!")
# print(type(tmpnodes_per_step))
# print(tmpnodes_per_step)