import osmnx as ox
import geopy.distance
import time
from KDtree import KDTree

G=ox.graph_from_address('Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México', dist=10000, network_type='drive')
ox.plot_graph(G)

id=1176137044
print('Origin node: '+str(G.nodes[id]))
print('Successor nodes: ')

for node in G.successors(id):
    print(node,G.nodes[node])
try:
    G2=ox.speed.add_edge_speeds(G)
    G3=ox.speed.add_edge_travel_times(G2)
except AttributeError:
    G2=ox.add_edge_speeds(G)
    G3=ox.add_edge_travel_times(G2)
buildings=ox.features_from_address('Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México', tags={'building':True}, dist=100)
print(buildings)

orig_node=1176137044
dest_node=1176149187
coord_orig=(G.nodes[orig_node]['y'],G.nodes[orig_node]['x'])
coord_dest=(G.nodes[dest_node]['y'],G.nodes[dest_node]['x'])
dist=geopy.distance.distance(coord_orig,coord_dest).m
print(dist)

points=[(data['x'],data['y']) for nid,data in G.nodes(data=True)]
t0=time.perf_counter()
kdtree=KDTree(points)
t1=time.perf_counter()
print(f"KD tree build time: {t1-t0:.6f} seconds")

query_locations=[
(20.7299,-103.4396),(20.7300,-103.4397),(20.7290,-103.4400),(20.7285,-103.4390),
(20.7310,-103.4385),(20.7280,-103.4410),(20.7305,-103.4405),(20.7275,-103.4380),
(20.7320,-103.4395),(20.7295,-103.4375),(20.7302,-103.4412),(20.7268,-103.4398),
(20.7292,-103.4420),(20.7315,-103.4415),(20.7288,-103.4370),(20.7330,-103.4400),
(20.7270,-103.4418),(20.7297,-103.4368),(20.7308,-103.4378),(20.7283,-103.4425),
]

for i,(qlat,qlon) in enumerate(query_locations,start=1):
    target=(qlon,qlat)
    s0=time.perf_counter()
    kp,kd_euclid=kdtree.nearest(target)
    s1=time.perf_counter()
    kd_ms=(s1-s0)*1000.0
    b0=time.perf_counter()
    best=None
    best_d2=float('inf')

    for p in points:
        dx=p[0]-target[0]
        dy=p[1]-target[1]
        d2=dx*dx+dy*dy
        if d2<best_d2:
            best_d2=d2
            best=p

    b1=time.perf_counter()
    brute_ms=(b1-b0)*1000.0
    kd_m=geopy.distance.distance((qlat,qlon),(kp[1],kp[0])).m if kp else None
    bf_m=geopy.distance.distance((qlat,qlon),(best[1],best[0])).m if best else None

    print(f"({i}): ({qlat:.3f},{qlon:.3f}) | KD: {kd_ms:.3f}ms | BF: {brute_ms:.3f}ms | KD dist={kd_m:.1f}m | BF dist={bf_m:.1f}m")