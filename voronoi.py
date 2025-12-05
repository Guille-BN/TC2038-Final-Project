import osmnx as ox
import geopy.distance
import time
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
address='Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México'
G=ox.graph_from_address(address,dist=10000,network_type='drive')
tags={'amenity':True,'healthcare':True}
pois=ox.features_from_address(address,tags=tags,dist=10000)
mask=[]
for idx,row in pois.iterrows():
    a=row.get('amenity')
    h=row.get('healthcare')
    if isinstance(a,str) and a.lower() in ('hospital','clinic','doctors'):
        mask.append(idx)
    elif isinstance(h,str) and h.lower() in ('hospital','clinic'):
        mask.append(idx)
pois=pois.loc[mask]
hospitals=[]
for idx,row in pois.iterrows():
    geom=row.geometry
    if geom is None:
        continue
    pt=geom.centroid if not geom.geom_type=='Point' else geom
    hospitals.append({'name':row.get('name',''), 'lat':pt.y, 'lon':pt.x})
if not hospitals:
    print('no hospitals found')
    raise SystemExit
lon_list=[h['lon'] for h in hospitals]
lat_list=[h['lat'] for h in hospitals]
hospital_nodes=ox.distance.nearest_nodes(G,lon_list,lat_list)
for i,h in enumerate(hospitals, start=1):
    print(f"{i}: name={h['name'] or 'N/A'} lat={h['lat']:.6f} lon={h['lon']:.6f} node={hospital_nodes[i-1]}")
G_proj=ox.project_graph(G)
node_list=list(G_proj.nodes(data=True))
node_coords=np.array([[d['x'],d['y']] for _,d in node_list])
hosp_proj=[]
for lon,lat in zip(lon_list,lat_list):
    p=Point(lon,lat)
    p_proj,crs=ox.projection.project_geometry(p,to_crs=G_proj.graph['crs'])
    hosp_proj.append((p_proj.x,p_proj.y))
hosp_proj_np=np.array(hosp_proj)
tree=cKDTree(hosp_proj_np)
dists,idxs=tree.query(node_coords,k=1)
assignments=idxs
counts=np.bincount(assignments,minlength=len(hospitals))
for i,c in enumerate(counts, start=1):
    print(f"hospital {i} assigned nodes={c}")
colors=plt.cm.tab20(np.linspace(0,1,len(hospitals)))
node_colors=[colors[a%len(colors)] for a in assignments]
fig,ax=plt.subplots(figsize=(10,10))
xs=node_coords[:,0]
ys=node_coords[:,1]
ax.scatter(xs,ys,s=2,c=node_colors,alpha=0.8)
for i,(hx,hy) in enumerate(hosp_proj_np):
    ax.scatter(hx,hy,s=60,c=colors[i%len(colors)],edgecolor='k')
    ax.text(hx,hy,str(i+1),color='k',fontsize=9,ha='center',va='center')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('voronoi_partition.png',dpi=200)
print('voronoi_partition.png saved')