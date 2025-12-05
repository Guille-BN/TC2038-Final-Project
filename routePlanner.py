import osmnx as ox
import geopy.distance
import time
from simpleai.search import SearchProblem,breadth_first,depth_first,uniform_cost,iterative_limited_depth_first,astar
iterative_deepening_depth_first = iterative_limited_depth_first

G=ox.graph_from_address('Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México',dist=10000,network_type='drive')
try:
    G2=ox.speed.add_edge_speeds(G)
    G3=ox.speed.add_edge_travel_times(G2)
except:
    G2=ox.add_edge_speeds(G)
    G3=ox.add_edge_travel_times(G2)
G=G3
nodes=list(G.nodes())

def dist(u,v):
    a=(G.nodes[u]['y'],G.nodes[u]['x'])
    b=(G.nodes[v]['y'],G.nodes[v]['x'])
    return geopy.distance.distance(a,b).m

pairs_short=[]
pairs_mid=[]
pairs_long=[]

for i in range(len(nodes)):
    for j in range(i+1,len(nodes)):
        d=dist(nodes[i],nodes[j])
        if d<=1000 and len(pairs_short)<5:
            pairs_short.append((nodes[i],nodes[j]))
        elif 1000<d<=5000 and len(pairs_mid)<5:
            pairs_mid.append((nodes[i],nodes[j]))
        elif d>5000 and len(pairs_long)<5:
            pairs_long.append((nodes[i],nodes[j]))
        if len(pairs_short)==5 and len(pairs_mid)==5 and len(pairs_long)==5:
            break
    if len(pairs_short)==5 and len(pairs_mid)==5 and len(pairs_long)==5:
        break

class RouteProblem(SearchProblem):
    def __init__(self,start,goal):
        super().__init__(start)
        self.goal=goal
    def actions(self,state):
        return list(G.successors(state))
    def result(self,state,action):
        return action
    def is_goal(self,state):
        return state==self.goal
    def cost(self,state1,action,state2):
        return G[state1][state2][0].get('travel_time',1)
    def heuristic(self,state):
        a=(G.nodes[state]['y'],G.nodes[state]['x'])
        b=(G.nodes[self.goal]['y'],G.nodes[self.goal]['x'])
        return geopy.distance.distance(a,b).m

def run_bfs(u,v):
    p=RouteProblem(u,v)
    t0=time.perf_counter()
    breadth_first(p,graph_search=True)
    t1=time.perf_counter()
    return t1-t0

def run_dfs(u,v):
    p=RouteProblem(u,v)
    t0=time.perf_counter()
    depth_first(p,graph_search=True)
    t1=time.perf_counter()
    return t1-t0

def run_ucs(u,v):
    p=RouteProblem(u,v)
    t0=time.perf_counter()
    uniform_cost(p,graph_search=True)
    t1=time.perf_counter()
    return t1-t0

max_depth = 1000  

def run_iddfs(u,v):
    p=RouteProblem(u,v)
    t0=time.perf_counter()
    try:
        iterative_deepening_depth_first(p,depth_limit=max_depth)
        t1=time.perf_counter()
        return f"{(t1-t0):.4f}s"
    except:
        return "Time out"

def run_ast(u,v):
    p=RouteProblem(u,v)
    t0=time.perf_counter()
    astar(p,graph_search=True)
    t1=time.perf_counter()
    return t1-t0

def print_results(title,pairs):
    print(title)
    for u,v in pairs:
        print(f"{u}->{v} {dist(u,v):.1f}m BFS:{run_bfs(u,v):.4f}s")
    print()
    for u,v in pairs:
        print(f"{u}->{v} {dist(u,v):.1f}m DFS:{run_dfs(u,v):.4f}s")
    print()
    for u,v in pairs:
        print(f"{u}->{v} {dist(u,v):.1f}m UCS:{run_ucs(u,v):.4f}s")
    print()
    for u,v in pairs:
        print(f"{u}->{v} {dist(u,v):.1f}m IDDFS:{run_iddfs(u,v)}")
    print()
    for u,v in pairs:
        print(f"{u}->{v} {dist(u,v):.1f}m A*:{run_ast(u,v):.4f}s")
    print()

print_results("<=1000m",pairs_short)
print_results("1000-5000m",pairs_mid)
print_results(">5000m",pairs_long)