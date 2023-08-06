from py2neo import *
from graphdatascience import GraphDataScience

# 4.1.1
graph = Graph('bolt://localhost:7687', auth=("neo4j", "zichaol"))
nodeMatcher = NodeMatcher(graph)
relationMatcher = RelationshipMatcher(graph)


def timespace(relation):
    time = relation.setdefault('time')
    target = relation.end_node.identity

    return time, target


def get_cij(id_i, id_j):
    p = 0
    count = 0
    node_i = nodeMatcher[id_i]
    node_j = nodeMatcher[id_j]
    rela_i = relationMatcher.match({node_i}, 'Locate').limit(None)
    rela_j = relationMatcher.match({node_j}, 'Locate').limit(None)
    for i in rela_i:
        time_i, space_i = timespace(i)
        for j in rela_j:
            time_j, space_j = timespace(j)
            if space_j == space_i:
                p = 1
                count += 1
    if p == 1:
        cij = 0.1 * count
    else:
        cij = 0
    return cij


def get_lij(id_i, id_j):
    node_i = nodeMatcher[id_i]
    node_j = nodeMatcher[id_j]
    rela_1 = relationMatcher.match({node_i, node_j}, 'Lineage').exists()  # 双向
    rela_2 = relationMatcher.match({node_j, node_i}, 'Lineage').exists()
    if not rela_1 and not rela_2:
        return 0
    else:
        return 0.3


def get_sij(id_i, id_j):
    lij = get_lij(id_i, id_j)
    cij = get_cij(id_i, id_j)
    # print("cij lij",cij,lij)
    if (lij + cij) == 0:
        return 0
    else:
        return 2 * lij * cij / (lij + cij)


place = graph.nodes.match('Place')  # 地点id
place_id = []
person = graph.nodes.match('Person')  # 人物id
person_id = []

for i in person:
    person_id.append(i.identity)
for i in place:
    place_id.append(i.identity)
id = []  # 全部节点id
nodes = graph.nodes
for i in nodes:
    id.append(i)

# 往图里加入权重weight
for i in range(len(id)):
    id_i = id[i]
    for j in range(len(id)):
        if i == j:  # 不对自己节点做处理
            continue
        id_j = id[j]
        node_i = nodeMatcher[id_i]
        node_j = nodeMatcher[id_j]

        if id_i in place_id or id_j in place_id:  # 有一个为地点节点，则为locate关系
            r = relationMatcher.match({node_i, node_j}).first()
            if r is not None:
                r.update({'weight': 0.2})
                graph.push(r)
        else:  # 人与人间的关系
            r = relationMatcher.match({node_i, node_j})
            for r1 in r:
                if r1 is not None:
                    r1.update({'weight': 0.8})
                    graph.push(r1)

for i in range(len(place_id)):  # 对地点节点做处理
    id_i = id[i]
    node_i = nodeMatcher[id_i]
    r = relationMatcher.match({node_i}).limit(None)  # 挑选出与地点节点相连的关系
    if r is not None:
        if len(r) >= 2:
            for j in r:
                j.update({'weight': 0.4})
                graph.push(j)

r = relationMatcher.match(r_type='Lineage').limit(None)
for r1 in r:
    node1, node2 = r1.nodes
    sij = get_sij(node1.identity, node2.identity)
    r1.update({'sij': sij})
    # print("aaaa")
    graph.push(r1)

# 加权PR计算，结果写入txt
import numpy as np
import random
import sys  # 导入sys包

np.set_printoptions(threshold=sys.maxsize)
epoch = 9


def create_data(N, epoch, alpha=0.5):  # random > alpha, then here is a edge.
    G = np.zeros((epoch, N))
    for i in range(N):
        if random.random() < alpha:
            G[0][1] = 1
    return G


PR = create_data(len(nodes), epoch)


def getPR(PR, epoch, node_id):
    d = 0.85

    epoch1 = 1
    while epoch1 <= epoch:
        for id1 in node_id:
            PR1 = 1 - d
            node = nodeMatcher[id1]
            rela = relationMatcher.match({node}).limit(None)  # 与节点A所有相连的rela
            for i in rela:
                target = i.end_node  # 节点Ti
                # print(target.identity)
                w = i['weight']
                c = relationMatcher.match({target}).count()
                PR1 += d * w * PR[epoch1 - 1][node_id.index(target.identity)] / c
            PR[epoch1][node_id.index(id1)] = PR1
        epoch1 += 1
    return PR


PR = getPR(PR, epoch - 1, id)
with open("1.txt", "w+")as f:
    f.write(str(PR))
    f.close()

for i in range(len(id)):
    id_i = id[i]
    PRi = PR[epoch - 1][i]
    nodei = nodeMatcher[id_i]
    nodei.update({'PR': float(PRi)})
    graph.push(nodei)

# 4.1.2 - 4.1.3
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "zichaol"))

if gds.graph.exists('my_graph').values[1]:
    gds.graph.drop(gds.graph.get('my_graph'))

graph.run(
    """CALL gds.graph.project('my_graph', ['Person', 'Place'],
        {Lineage:{orientation: 'UNDIRECTED'}, Locate:{orientation: 'UNDIRECTED'}},
        {relationshipProperties: 'weight'})"""
)

# Louvain

graph.run(
    """CALL gds.louvain.write('my_graph', { writeProperty: 'NoneWeightedCommunity' })
        YIELD communityCount, modularity, modularities"""
)

graph.run(
    """CALL gds.louvain.write('my_graph', { relationshipWeightProperty: 'weight',writeProperty: 'WeightedCommunity' })
        YIELD communityCount, modularity, modularities"""
)

# PageRank

graph.run(
    """CALL gds.pageRank.write('my_graph', {
        maxIterations: 20,
        dampingFactor: 0.85,
        writeProperty: 'pagerank'
        })
        YIELD nodePropertiesWritten, ranIterations"""
)
