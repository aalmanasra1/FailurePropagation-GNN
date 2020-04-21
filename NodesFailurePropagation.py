# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets.demos import models
from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
import copy
import networkx as nx

try:
  import seaborn as sns
except ImportError:
  pass
else:
  sns.reset_orig()

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

def base_graph(n, d):
  print("base_graph")
  nodes = np.zeros((n, 5), dtype=np.float32)
  half_width = d * n / 2.0
  nodes[:, 0] = np.linspace(
      -half_width, half_width, num=n, endpoint=False, dtype=np.float32)
  # indicate that the first and last masses are fixed
  nodes[(0, -1), -1] = 1.

  # Edges.
  edges, senders, receivers = [], [], []
  for i in range(n - 1):
    left_node = i
    right_node = i + 1
    # The 'if' statements prevent incoming edges to fixed ends of the string.
    if right_node < n - 1:
      # Left incoming edge.
      edges.append([50., d])
      senders.append(left_node)
      receivers.append(right_node)
    if left_node > 0:
      # Right incoming edge.
      edges.append([50., d])
      senders.append(right_node)
      receivers.append(left_node)

  return {
      "globals": [0., -10.],
      "nodes": nodes,
      "edges": edges,
      "receivers": receivers,
      "senders": senders
  }

#@title Helper functions  { form-width: "30%" }
# pylint: disable=redefined-outer-name
def getWhiteNodesIndeces(nodes):
  i=0
  white_nodes_indeces=[]
  for n in nodes:
    if n[0]==0:
      white_nodes_indeces.append(i)
    i+=1
  return white_nodes_indeces


def plot_graphs_tuple_np(graphs_tuple,phase):
    graphs_nx = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
    fig, axs = plt.subplots(ncols=10, figsize=(20, 2))
    for iax, (graph_nx, ax) in enumerate(zip(graphs_nx, axs)):
        graph_t = utils_np.networkxs_to_graphs_tuple([graph_nx])
        graph_d = utils_np.graphs_tuple_to_data_dicts(graph_t)
        # print(type(graph_d))
        nodes = graph_d[0]["nodes"]
        black_nodes = getBlackNodesIndecesPrediction(nodes)
        white_nodes = getWhiteNodesIndeces(nodes)
        print("black_nodes")
        print(black_nodes)
        print("white_nodes")
        print(white_nodes)
        color_map = []
        for i in range(len(nodes)):
            if i in black_nodes:
                color_map.append('r')
            else:
                color_map.append('g')
        print("color_map")
        print(color_map)
        pos = nx.spring_layout(graph_nx)
        pos = {0: (10, 20), 1: (20, 30), 2: (30, 40), 3: (50, 60), 4: (50, 70), 5: (50, 80), 6: (50, 90), 7: (40, 70),
               8: (40, 80)}
        nx.draw(graph_nx,pos = {0: (10, 20), 1: (20, 30), 2: (30, 40), 3: (30, 50), 4: (50, 50), 5: (60, 60), 6: (70, 70), 7: (80, 80),
               8: (40, 80)}, ax=ax, node_color=color_map,
                node_size=100,
                alpha=0.8)
        # nx.draw_networkx_nodes(graph_nx,pos, ax=ax, nodelist=white_nodes,
        #         node_color='g',
        #         node_size=100,
        #         alpha=0.8)
        # if phase == 1:
        #     ax.set_title("Step {}".format(iax))
        # else:
        #     x = iax
        #     ax.set_title("Step {}".format(x+5))
        ax.set_title("Step {}".format(iax))
        if phase == 1:
            # fig.suptitle('True trajectory', fontsize=10)
            fig = plt.gcf()
            fig.canvas.set_window_title('True trajectory')
        else:
            # fig.suptitle('Predicted trajectory', fontsize=10)
            fig = plt.gcf()
            fig.canvas.set_window_title('Predicted trajectory')
        nodes = []
        black_nodes = []
        white_nodes = []


def mybase_graph_ge(n):
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
  # [isHealthy, RemainingTime, 500_count, CPU_thread_count, Memory_utilization, Cpu_utilization]

  globals_2 = [2., 3., 4.]
  
  # nodes_2 = [[0.,0.],  # Node 0
  #          [0.,0.],  # Node 1
  #          [0.,0.],  # Node 2
  #          [0.,0.],  # Node 3
  #          [0.,0.],  # Node 4
  #          [0.,0.],  # Node 5
  #          [0.,0.],  # Node 6
  #          [0.,0.],  # Node 7
  #          [0.,0.]]  # Node 8

  nodes_2 = [[0., 0.],  # Node 0
             [0., 0.],  # Node 1
             [0., 0.],  # Node 2
             [0., 0.],  # Node 3
             [0., 0.]]  # Node 4

  for index, value in enumerate(nodes_2):
      if index == n:
        nodes_2[index][0] = 1.
        nodes_2[index][1] = 1.

  # # Edge features for graph 0.
  # edges_2 = [[10., 20.],  # Edge 0
  #           [11., 21.],  # Edge 1
  #           [12., 22.],  # Edge 2
  #           [13., 23.],  # Edge 3
  #           [14., 24.],  # Edge 4
  #           [14., 24.],  # Edge 5
  #           [14., 24.],  # Edge 6
  #           [15., 25.]]  # Edge 7
  #
  # # The sender and receiver nodes associated with each edge for graph 0.
  # senders_2 = [0,  # Index of the sender node for edge 0
  #             1,  # Index of the sender node for edge 1
  #             2,  # Index of the sender node for edge 2
  #             5,  # Index of the sender node for edge 3
  #             8,  # Index of the sender node for edge 4
  #             5,  # Index of the sender node for edge 5
  #             5,  # Index of the sender node for edge 6
  #             7]  # Index of the sender node for edge 7
  # receivers_2 = [1,  # Index of the receiver node for edge 0
  #               3,  # Index of the receiver node for edge 1
  #               4,  # Index of the receiver node for edge 2
  #               3,  # Index of the receiver node for edge 3
  #               6,  # Index of the receiver node for edge 4
  #               6,  # Index of the receiver node for edge 5
  #               2,  # Index of the receiver node for edge 6
  #               3]  # Index of the sender node for edge 7

  # Edge features for graph 0.
  edges_2 = [[10., 20.],  # Edge 0
             [11., 21.],  # Edge 1
             [12., 22.],  # Edge 2
             [13., 23.]]  # Edge 3
  # The sender and receiver nodes associated with each edge for graph 0.
  senders_2 = [0,  # Index of the sender node for edge 0
               1,  # Index of the sender node for edge 1
               2,  # Index of the sender node for edge 2
               3]  # Index of the sender node for edge 3
  receivers_2 = [1,  # Index of the receiver node for edge 0
                 2,  # Index of the receiver node for edge 1
                 4,  # Index of the receiver node for edge 2
                 1]  # Index of the sender node for edge 3
  data_dict_2 = {
    "globals": globals_2,
    "nodes": nodes_2,
    "edges": edges_2,
    "senders": senders_2,
    "receivers": receivers_2
  }
  return data_dict_2

def mybase_graph(n,graph_shape_dist):
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


  if graph_shape_dist <= 80:
      nodes_1 = [[0.,0.],  # Node 0
               [0.,0.],  # Node 1
               [0.,0.],  # Node 2
               [0.,0.],  # Node 3
               [0.,0.]]  # Node 4
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
                   2,  # Index of the sender node for edge 4
                   3]  # Index of the sender node for edge 5
      receivers_1 = [1,  # Index of the receiver node for edge 0
                     2,  # Index of the receiver node for edge 1
                     3,  # Index of the receiver node for edge 2
                     0,  # Index of the receiver node for edge 3
                     4,  # Index of the sender node for edge 4
                     4]  # Index of the receiver node for edge 5
      nodes_1[n%5][0] = 1.
      nodes_1[n%5][1] = 1.
  elif graph_shape_dist <= 160:
      nodes_1 = np.zeros((6, 2), dtype=np.float32)
      edges_1 = [[100., 200.],  # Edge 0
                 [101., 201.],  # Edge 1
                 [102., 202.],  # Edge 2
                 [103., 203.],  # Edge 3
                 [104., 204.],
                 [104., 204.],# Edge 4
                 [105., 205.]]  # Edge 5

      # The sender and receiver nodes associated with each edge for graph 0.
      senders_1 = [0,  # Index of the sender node for edge 0
                   1,  # Index of the sender node for edge 1
                   1,  # Index of the sender node for edge 2
                   2,  # Index of the sender node for edge 3
                   2,
                   5,# Index of the sender node for edge 4
                   3]  # Index of the sender node for edge 5
      receivers_1 = [1,  # Index of the receiver node for edge 0
                     2,  # Index of the receiver node for edge 1
                     3,  # Index of the receiver node for edge 2
                     5,  # Index of the receiver node for edge 3
                     4,
                     0,# Index of the sender node for edge 4
                     4]  # Index of the receiver node for edge 5
      nodes_1[n % 6][0] = 1.
      nodes_1[n % 6][1] = 1.

  elif graph_shape_dist <= 240:
      nodes_1 = np.zeros((7, 2), dtype=np.float32)
      edges_1 = [[100., 200.],  # Edge 0
                 [101., 201.],  # Edge 1
                 [102., 202.],  # Edge 2
                 [103., 203.],  # Edge 3
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],# Edge 4
                 [105., 205.]]  # Edge 5

      # The sender and receiver nodes associated with each edge for graph 0.
      senders_1 = [0,  # Index of the sender node for edge 0
                   1,  # Index of the sender node for edge 1
                   1,  # Index of the sender node for edge 2
                   2,  # Index of the sender node for edge 3
                   4,
                   6,
                   5,# Index of the sender node for edge 4
                   3]  # Index of the sender node for edge 5
      receivers_1 = [1,  # Index of the receiver node for edge 0
                     2,  # Index of the receiver node for edge 1
                     3,  # Index of the receiver node for edge 2
                     0,  # Index of the receiver node for edge 3
                     2,
                     5,
                     4,# Index of the sender node for edge 4
                     4]  # Index of the receiver node for edge 5
      nodes_1[n % 7][0] = 1.
      nodes_1[n % 7][1] = 1.

  elif graph_shape_dist <= 320:
      nodes_1 = np.zeros((8, 2), dtype=np.float32)
      edges_1 = [[100., 200.],  # Edge 0
                 [101., 201.],  # Edge 1
                 [102., 202.],  # Edge 2
                 [103., 203.],  # Edge 3
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],# Edge 4
                 [105., 205.]]  # Edge 5

      # The sender and receiver nodes associated with each edge for graph 0.
      senders_1 = [0,  # Index of the sender node for edge 0
                   1,  # Index of the sender node for edge 1
                   1,  # Index of the sender node for edge 2
                   2,  # Index of the sender node for edge 3
                   2,
                   3,
                   3,
                   3,
                   7,
                   5,# Index of the sender node for edge 4
                   3]  # Index of the sender node for edge 5
      receivers_1 = [1,  # Index of the receiver node for edge 0
                     2,  # Index of the receiver node for edge 1
                     3,  # Index of the receiver node for edge 2
                     0,  # Index of the receiver node for edge 3
                     4,
                     7,
                     6,
                     5,
                     6,
                     6,# Index of the sender node for edge 4
                     4]  # Index of the receiver node for edge 5
      nodes_1[n % 8][0] = 1.
      nodes_1[n % 8][1] = 1.

  else:
      nodes_1 = np.zeros((9, 2), dtype=np.float32)
      edges_1 = [[100., 200.],  # Edge 0
                 [101., 201.],  # Edge 1
                 [102., 202.],  # Edge 2
                 [103., 203.],  # Edge 3
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],
                 [104., 204.],# Edge 4
                 [105., 205.]]  # Edge 5

      # The sender and receiver nodes associated with each edge for graph 0.
      senders_1 = [0,  # Index of the sender node for edge 0
                   1,  # Index of the sender node for edge 1
                   3,  # Index of the sender node for edge 2
                   2,  # Index of the sender node for edge 3
                   4,
                   4,
                   5,
                   5,
                   7,
                   7,
                   8,# Index of the sender node for edge 4
                   3]  # Index of the sender node for edge 5
      receivers_1 = [1,  # Index of the receiver node for edge 0
                     2,  # Index of the receiver node for edge 1
                     1,  # Index of the receiver node for edge 2
                     0,  # Index of the receiver node for edge 3
                     2,
                     5,
                     7,
                     8,
                     8,
                     6,
                     2,
                     # Index of the sender node for edge 4
                     4]  # Index of the receiver node for edge 5
      nodes_1[n][0] = 1.
      nodes_1[n][1] = 1.

  # nodes_1 = [[0., 1., 1., 1., 1.],  # Node 0
  #            [0., 0., 1., 1., 1.],  # Node 1
  #            [0., 0., 1., 1., 1.],  # Node 2
  #            [0., 0., 1., 1., 1.],  # Node 3
  #            [0., 0., 1., 1., 1.],  # Node 4
  #            [0., 0., 1., 1., 1.],  # Node 5
  #            [0., 0., 1., 1., 1.],  # Node 6
  #            [0., 0., 1., 1., 1.],  # Node 7
  #            [0., 0., 1., 1., 1.],  # Node 8
  #            [0., 0., 1., 1., 1.],  # Node 9
  #            [0., 0., 1., 1., 1.]]  # Node 10
  
  # for index, value in enumerate(nodes_1):
  #     if index == (n%5):
  #       nodes_1[index][0] = 1.
  #       nodes_1[index][1] = 1.
  

  
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
def getBlackNodesIndecesPrediction(nodes):
  i=0
  black_nodes_indeces=[]
  for n in nodes:
    if 0.90 <= n[0] :
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
    graph: grap as a python dictionary 

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

def prediction_to_next_state(input_graph, predicted_graph, step_size):
  print("prediction_to_next_state")
  # manually integrate velocities to compute new positions
  # new_pos = input_graph.nodes[..., : 2] + predicted_graph.nodes * step_size
  new_pos = predicted_graph.nodes
  print("new_pos")
  print(new_pos)
  # new_nodes = tf.concat(
  #     [new_pos, predicted_graph.nodes, input_graph.nodes[..., 0:2]], axis=-1)
  new_nodes = predicted_graph.nodes[..., 0:2]
  # new_nodes = tf.concat(
  #     [new_pos, input_graph.nodes[..., 1:2]], axis=-1)
  print("new_nodes")
  print(new_nodes)
  return input_graph.replace(nodes=new_nodes)

def roll_out_myphysics(simulator, graph, steps, step_size):
  PredictedGraphSteps=[]
  nodes_per_step = tf.TensorArray(
      dtype=graph.nodes.dtype, size=steps + 1, element_shape=graph.nodes.shape)
  nodes_per_step = nodes_per_step.write(0, graph.nodes)
  PredictedGraphSteps.append(graph)

  for t in range(1,steps+1):
      predicted_graph = simulator(graph)
      if isinstance(predicted_graph, list):
          predicted_graph = predicted_graph[-1]
      graph = prediction_to_next_state(graph, predicted_graph, step_size)
      PredictedGraphSteps.append(graph)
      print("PredictedGraphSteps appended " + str(t))
      nodes_per_step = nodes_per_step.write(t, graph.nodes)

  return graph, nodes_per_step.stack(), PredictedGraphSteps

def generate_blackwhite_trajectory(bw_static_graph_tr, steps, step_size):
  # print("<---generate_blackwhite_trajectory--->")
  mybase_graph_tr = utils_tf.data_dicts_to_graphs_tuple(bw_static_graph_tr)



  # List of lists: it contains the status of 200 graphs over 50 steps
  ListofGraphSteps=[]

  c=0
  for static_graph in bw_static_graph_tr:
    graph = static_graph
    graph_steps=[]
    graph_steps.append(graph)
    for i in range(1,steps+1):
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
  tmpfilldictionary=[]
  for tmpgs in ListofGraphSteps:
      tmpfilldictionary.append(tmpgs[k])
      ListofGraphStepsCounter=ListofGraphStepsCounter+1

  tmpgraph_steps_tuple=utils_tf.data_dicts_to_graphs_tuple(tmpfilldictionary)
  tmpnodes_per_step = tf.TensorArray(
        dtype=tmpgraph_steps_tuple.nodes.dtype, size=steps+1, element_shape=tmpgraph_steps_tuple.nodes.shape)

  gscount=0
  for i in range(steps+1):
    filldictionary=[]
    for gs in ListofGraphSteps:
      filldictionary.append(gs[i])
      gscount=gscount+1
    gscount=0
    tmpnodes_per_step = tmpnodes_per_step.write(i, utils_tf.data_dicts_to_graphs_tuple(filldictionary).nodes)

  return mybase_graph_tr, tmpnodes_per_step.stack(), ListofGraphSteps

def create_loss_ops(target_op, output_ops):
  print("create_loss_ops")
  """Create supervised loss operations from targets and outputs.

  Args:
    target_op: The target velocity tf.Tensor.
    output_ops: The list of output graphs from the model.

  Returns:
    A list of loss values (tf.Tensor), one per output op.
  """
  # target_op[..., 2:4] = tf.Print(target_op[..., 2:4], [target_op[..., 2:4]], message="target_op[..., 2:4]: ")
  # loss_ops = [
  #     tf.reduce_mean(
  #         tf.reduce_sum((output_op.nodes[..., 0:1] - target_op[..., 0:1])**2, axis=-1))
  #     for output_op in output_ops
  # ]
  # isHealthy_diff_count=0
  # for node,target_node in zip(output_ops[0].nodes,target_op):
  #     res=node[..., 0:1]-target_node[..., 0:1]
  #     if res != 0:
  #         isHealthy_diff_count+=1
  # isHealthy_error = [
  #     tf.reduce_mean(
  #         tf.reduce_sum((output_op.nodes[..., 0:1] - target_op[..., 0:1])**2, axis=-1))
  #     for output_op in output_ops
  # ]
  # RT_error = [
  #     tf.reduce_mean(
  #         tf.reduce_sum((output_op.nodes[..., 1:2] - target_op[..., 1:2]) ** 2, axis=-1))
  #     for output_op in output_ops
  # ]
  # loss_ops = isHealthy_error + RT_error
  # loss_ops = isHealthy_diff_count
  bce = tf.keras.losses.binary_crossentropy

  loss_ops = [
        bce(target_op[..., 0:2], output_op.nodes[..., 0:2])
      for output_op in output_ops
  ]
  return loss_ops


def make_all_runnable_in_session(*args):
  print("make_all_runnable_in_session")
  """Apply make_runnable_in_session to an iterable of graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


# pylint: enable=redefined-outer-name

#@title Set up model training and evaluation  { form-width: "30%" }

# The model we explore includes three components:
# - An "Encoder" graph net, which independently encodes the edge, node, and
#   global attributes (does not compute relations etc.).
# - A "Core" graph net, which performs N rounds of processing (message-passing)
#   steps. The input to the Core is the concatenation of the Encoder's output
#   and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
#   the processing step).
# - A "Decoder" graph net, which independently decodes the edge, node, and
#   global attributes (does not compute relations etc.), on each
#   message-passing step.
#
#                     Hidden(t)   Hidden(t+1)
#                        |            ^
#           *---------*  |  *------*  |  *---------*
#           |         |  |  |      |  |  |         |
# Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
#           |         |---->|      |     |         |
#           *---------*     *------*     *---------*
#
# The model is trained by supervised learning. Input mass-spring systems are
# procedurally generated, where the nodes represent the positions, velocities,
# and indicators of whether the mass is fixed in space or free to move, the
# edges represent the spring constant and spring rest length, and the global
# attribute represents the variable coefficient of gravitational acceleration.
# The outputs/targets have the same structure, with the nodes representing the
# masses' next-step states.
#
# The training loss is computed on the output of each processing step. The
# reason for this is to encourage the model to try to solve the problem in as
# few steps as possible. It also helps make the output of intermediate steps
# more interpretable.
#
# There's no need for a separate evaluate dataset because the inputs are
# never repeated, so the training loss is the measure of performance on graphs
# from the input distribution.
#
# We also evaluate how well the models generalize to systems which are one mass
# larger, and smaller, than those from the training distribution. The loss is
# computed as the mean over a 50-step rollout, where each step's input is the
# the previous step's output.
#
# Variables with the suffix _tr are training parameters, and variables with the
# suffix _ge are test/generalization parameters.
#
# After around 10000-20000 training iterations the model reaches good
# performance on mass-spring systems with 5-8 masses.

tf.reset_default_graph()

rand = np.random.RandomState(SEED)

# Model parameters.
num_processing_steps_tr = 1
num_processing_steps_ge = 1

# Data / training parameters.

num_training_iterations = 10000
num_time_steps = 10
step_size = 0.1

# Create the model.
# model = models.EncodeProcessDecode(node_output_size=2)
model = models.EncodeProcessDecode(node_output_size=2)
# model = models.MLPGraphNetwork()



######################My Work!########################
possible_black_node_index_tr = (0, 8)
possible_black_node_index_ge = (0, 5)
# possible_black_node_index_ge = (0, 9)
bw_batch_size_tr=400
bw_batch_size_ge=50
first_black_nodes_tr = rand.randint(*possible_black_node_index_tr, size=bw_batch_size_tr)
first_black_nodes_ge = rand.randint(*possible_black_node_index_ge, size=bw_batch_size_ge)

# training
bw_static_graph_tr = []
dist_count=0
for i in first_black_nodes_tr:
  bw_static_graph_tr.append(mybase_graph(i, dist_count))
  dist_count+=1


print("<---bw_static_graph_tr--->")
print(bw_static_graph_tr)
initial_conditions_tr, true_trajectory_tr, visualization_true_trajectory_tr= generate_blackwhite_trajectory(bw_static_graph_tr, num_time_steps, step_size)
print("<---right after generate_blackwhite_trajectory training--->")

# for example_number in range(0, 5):
#     test = utils_np.data_dicts_to_graphs_tuple(visualization_true_trajectory_tr[example_number])
#     first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_np,1)
#     first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_np,2)
#     plt.show()
#
# for example_number in range(90, 95):
#     test = utils_np.data_dicts_to_graphs_tuple(visualization_true_trajectory_tr[example_number])
#     first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_np,1)
#     first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_np,2)
#     plt.show()
#
# for example_number in range(170, 175):
#     test = utils_np.data_dicts_to_graphs_tuple(visualization_true_trajectory_tr[example_number])
#     first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_np,1)
#     first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_np,2)
#     plt.show()
#
# for example_number in range(260, 265):
#     test = utils_np.data_dicts_to_graphs_tuple(visualization_true_trajectory_tr[example_number])
#     first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_np,1)
#     first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_np,2)
#     plt.show()
#
# for example_number in range(390, 395):
#     test = utils_np.data_dicts_to_graphs_tuple(visualization_true_trajectory_tr[example_number])
#     first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_np,1)
#     first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_np,2)
#     plt.show()
# generalization
bw_static_graph_ge = []
for j in first_black_nodes_ge:
  bw_static_graph_ge.append(mybase_graph_ge(j))


# Training.


initial_conditions_9_ge, true_nodes_rollout_9_ge, visualization_true_nodes_rollout_9_ge = generate_blackwhite_trajectory(bw_static_graph_ge, num_time_steps, step_size)

test_initials , predicted_nodes_rollout_9_ge, visualization_predicted_nodes_rollout_9_ge = roll_out_myphysics(
    lambda x: model(x, num_processing_steps_ge), initial_conditions_9_ge,
    num_time_steps, step_size)



# for example_number in range(0, 5):
#     test = utils_np.data_dicts_to_graphs_tuple(visualization_true_nodes_rollout_9_ge[example_number])
#     first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_np,1)
#     first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_np,2)
#     plt.show()

print("len(visualization_true_nodes_rollout_9_ge))")
print(len(visualization_true_nodes_rollout_9_ge))
print("len(visualization_true_nodes_rollout_9_ge[5]))")
print(len(visualization_true_nodes_rollout_9_ge[5]))



# Test/generalization loss: 9-nodes.
loss_op_9_ge = tf.reduce_mean(
    tf.reduce_sum(
        (predicted_nodes_rollout_9_ge[..., 0:1] -
         true_nodes_rollout_9_ge[..., 0:1])**2,
        axis=-1))
print("<---loss_op_9_ge--->")
print(loss_op_9_ge)


t = tf.random_uniform([], minval=0, maxval=num_time_steps - 1, dtype=tf.int32)
input_graph_tr = initial_conditions_tr.replace(nodes=true_trajectory_tr[t])
print("<---input_graph_tr--->")
print(input_graph_tr)
target_nodes_tr = true_trajectory_tr[t + 1]
print("<---target_nodes_tr--->")
print(target_nodes_tr)

print("<---before calling model--->")
output_ops_tr = model(input_graph_tr, num_processing_steps_tr)
print("<---after calling model--->")

print("<---right after output_ops_tr--->")
print("<---output_ops_tr--->")
print(output_ops_tr)


# Training loss.
loss_ops_tr = create_loss_ops(target_nodes_tr, output_ops_tr)
# print("<---loss_ops_tr--->")
# print(loss_ops_tr)
# Training loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
print("<---loss_op_tr--->")
print(loss_op_tr)
# loss_op_tr = tf.Print(loss_op_tr, [loss_op_tr], message="This is loss_op_tr: ")


# Optimizer.
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

input_graph_tr = make_all_runnable_in_session(input_graph_tr)
print("make_all_runnable_in_session(input_graph_tr)")
initial_conditions_9_ge = make_all_runnable_in_session(initial_conditions_9_ge)
print("make_all_runnable_in_session(initial_conditions_9_ge)")

#@title Reset session  { form-width: "30%" }

# This cell resets the Tensorflow session, but keeps the same computational
# graph.

try:
  sess.close()
  print("sess.close()")
except NameError:
  pass
sess = tf.Session()

x = sess.run(tf.global_variables_initializer())
print("sess.run(tf.global_variables_initializer())")

# mygraph = tf.get_default_graph()
# myoperations= mygraph.get_operations()
# print("my_graph opertions:\n")
# for op in myoperations: print(op.name)

last_iteration = 0
logged_iterations = []
losses_tr = []
losses_9_ge = []

#@title Run training  { form-width: "30%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.

# How much time between logging and printing the current results.
log_every_seconds = 2

print("# (iteration number), T (elapsed seconds), "
      "Ltr (training 1-step loss), "
      "Lge8 (test/generalization rollout loss for 8 nodes)")

start_time = time.time()
last_log_time = start_time
# Solution starts from here! We need to train our model
# by calling some random state and the one after
# and compute the loss for a number of iterations (epochs) :D
# NIGHTY ;)

print(last_iteration,num_training_iterations)
for iteration in range(last_iteration, num_training_iterations):
  last_iteration = iteration
  train_values = sess.run({
      "step": step_op,
      "loss": loss_op_tr,
      "input_graph": input_graph_tr,
      "target_nodes": target_nodes_tr,
      "outputs": output_ops_tr
  })
  the_time = time.time()
  elapsed_since_last_log = the_time - last_log_time
  if elapsed_since_last_log > log_every_seconds:
    last_log_time = the_time
    test_values = sess.run({
      "loss_9": loss_op_9_ge,
        "true_rollout_9": true_nodes_rollout_9_ge,
        "predicted_rollout_9": predicted_nodes_rollout_9_ge
    })
    elapsed = time.time() - start_time
    losses_tr.append(train_values["loss"])
    losses_9_ge.append(test_values["loss_9"])
    logged_iterations.append(iteration)
    print(iteration,elapsed,train_values["loss"],test_values["loss_9"])
    print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge9 {:.4f}".format(
        iteration, elapsed, train_values["loss"], test_values["loss_9"]))

    
#@title Visualize loss curves  { form-width: "30%" }

# This cell visualizes the results of training. You can visualize the
# intermediate results by interrupting execution of the cell above, and running
# this cell. You can then resume training by simply executing the above cell
# again.

def get_node_trajectories(rollout_array, batch_size):
  print("get_node_trajectories")  # pylint: disable=redefined-outer-name
  return np.split(rollout_array[..., :2], batch_size, axis=1)


fig = plt.figure(1, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)
# Next-step Loss.
y = losses_tr
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y, "k")
ax.set_title("Training loss")

y = losses_9_ge
ax = fig.add_subplot(1, 3, 2)
ax.plot(x, y, "k")
ax.set_title("Generalization loss: 9 nodes")


# # Visualize trajectories.
true_rollouts_9 = get_node_trajectories(test_values["true_rollout_9"],
                                        bw_batch_size_ge)
predicted_rollouts_9 = get_node_trajectories(test_values["predicted_rollout_9"],
                                             bw_batch_size_ge)

print("visualization_true_nodes_rollout_9_ge")
print(visualization_true_nodes_rollout_9_ge)

print("visualization_predicted_nodes_rollout_9_ge")
print(visualization_predicted_nodes_rollout_9_ge)



listofgraphsteps=[]
for i in range(0,num_time_steps+1):
    runnable = make_all_runnable_in_session(visualization_predicted_nodes_rollout_9_ge[i])
    visualization_predicted_nodes_rollout_9_ge_data_np = sess.run(runnable)
    listofgraphsteps.append(visualization_predicted_nodes_rollout_9_ge_data_np)


# prediction_over_time=[]
w, h = num_time_steps+1, bw_batch_size_ge;
prediction_over_time = [[0 for x in range(w)] for y in range(h)]
for k in range(0,num_time_steps+1):
    step_graphs = utils_np.graphs_tuple_to_data_dicts(listofgraphsteps[k][0])
    for j in range(0,bw_batch_size_ge):
        prediction_over_time[j][k]=step_graphs[j]

print("len(prediction_over_time))")
print(len(prediction_over_time))
print("len(prediction_over_time[5]))")
print(len(prediction_over_time[5]))

# for example_number in range(0,bw_batch_size_ge): # all examples
# for example_number in range(0, 5):  # first 10 examples
#     test2 = utils_np.data_dicts_to_graphs_tuple(prediction_over_time[example_number])
#     first_six_graphs_pred_np = utils_np.get_graph(test2, slice(0, 5))
#     plot_graphs_tuple_np(first_six_graphs_pred_np,1)
#     first_six_graphs_pred_np = utils_np.get_graph(test2, slice(5, 10))
#     plot_graphs_tuple_np(first_six_graphs_pred_np,2)
#     plt.show()


#         test = utils_np.data_dicts_to_graphs_tuple(visualization_true_nodes_rollout_9_ge[example_number])
#         first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#         plot_graphs_tuple_np(first_six_graphs_np,1)
#         first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#         plot_graphs_tuple_np(first_six_graphs_np,2)
#         plt.show()
for example_number in range(0, 5):  # first 10 examples
    test = utils_np.data_dicts_to_graphs_tuple(visualization_true_nodes_rollout_9_ge[example_number])
    test2 = utils_np.data_dicts_to_graphs_tuple(prediction_over_time[example_number])
    first_six_graphs_pred_np = utils_np.get_graph(test, slice(0, 10))
    plot_graphs_tuple_np(first_six_graphs_pred_np, 1)
    first_six_graphs_pred_np = utils_np.get_graph(test2, slice(0, 10))
    plot_graphs_tuple_np(first_six_graphs_pred_np,2)
    plt.show()

for example_number in range(15, 25):  # first 10 examples
    test = utils_np.data_dicts_to_graphs_tuple(visualization_true_nodes_rollout_9_ge[example_number])
    test2 = utils_np.data_dicts_to_graphs_tuple(prediction_over_time[example_number])
    first_six_graphs_pred_np = utils_np.get_graph(test, slice(0, 10))
    plot_graphs_tuple_np(first_six_graphs_pred_np, 1)
    first_six_graphs_pred_np = utils_np.get_graph(test2, slice(0, 10))
    plot_graphs_tuple_np(first_six_graphs_pred_np, 2)
    plt.show()
for example_number in range(40, 50):  # first 10 examples
    test = utils_np.data_dicts_to_graphs_tuple(visualization_true_nodes_rollout_9_ge[example_number])
    test2 = utils_np.data_dicts_to_graphs_tuple(prediction_over_time[example_number])
    first_six_graphs_pred_np = utils_np.get_graph(test, slice(0, 10))
    plot_graphs_tuple_np(first_six_graphs_pred_np, 1)
    first_six_graphs_pred_np = utils_np.get_graph(test2, slice(0, 10))
    plot_graphs_tuple_np(first_six_graphs_pred_np, 2)
    plt.show()


plt.show()

# def print_results():
#     for example_number in range(0, 5):
#         test = utils_np.data_dicts_to_graphs_tuple(visualization_true_nodes_rollout_9_ge[example_number])
#         first_six_graphs_np = utils_np.get_graph(test, slice(0, 5))
#         plot_graphs_tuple_np(first_six_graphs_np,1)
#         first_six_graphs_np = utils_np.get_graph(test, slice(5, 10))
#         plot_graphs_tuple_np(first_six_graphs_np,2)
#         plt.show()
#
#     for example_number in range(0, 5):  # first 10 examples
#         test2 = utils_np.data_dicts_to_graphs_tuple(prediction_over_time[example_number])
#         first_six_graphs_pred_np = utils_np.get_graph(test2, slice(0, 5))
#         plot_graphs_tuple_np(first_six_graphs_pred_np, 1)
#         first_six_graphs_pred_np = utils_np.get_graph(test2, slice(5, 10))
#         plot_graphs_tuple_np(first_six_graphs_pred_np,2)
#         plt.show()