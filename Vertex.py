import zen
import sys
import numpy
from numpy import *
import matplotlib.pyplot as plt
sys.path.append('../zend3js/')
import d3js
import string
import time
from time import sleep
import csv
numpy.set_printoptions(threshold=numpy.nan)
i=[]
j=[]
G=zen.Graph()
# with open('vertices_de_power_151109.csvdata','rb') as csvfile:
# 	a=csv.DictReader(csvfile)
# 	for row in a:
# 		i=(row['v_id'])
# 		G.add_node(i)

#print "Number of Nodes are: %i" %G.num_nodes
#print "Number of Edges are: %i" %G.num_edges
with open('generators.csv','rb') as csvfile:
	a=csv.DictReader(csvfile)
	for row in a:
		bus_no,carrier=(row['bus'],row['carrier'])
		bus_no=bus_no.split('_')

		if G.__contains__(bus_no[0]):
			b=G.node_data(bus_no[0])
			b=b+', '+carrier
			G.set_node_data(bus_no[0],b)
			continue
		G.add_node_x(int(bus_no[0]),1,bus_no[0],data=carrier)
x=G.nodes()
count=1
for i in x:
	if int(i)!=count:
		print i,count
		G.add_node_x(count,1,str(count),data='others')
		count=count+1
		if int(i)!=count:
			G.add_node_x(count,1,str(count),data='others')
			count=count+1
	count=count+1
# for i in x:
# 	print i,G.node_data(i)
with open('links_de_power_151109.csv','rb') as csvfile:
	a=csv.DictReader(csvfile)
	for row in a:
		i,j,k,voltage=(row['v_id_1'],row['v_id_2'],row['l_id'],row['voltage'])
		if int(voltage)>220001:
			volt_type='Industrial'
		volt_type='Domestic'
		if G.has_edge(i,j):
			 w=G.weight(i,j)+1
			 	# print i,j,w
			 G.set_weight(i,j,w)
			 G.set_edge_data(i,j,data=volt_type)
			 continue
		G.add_edge(j,i,data=volt_type)
N= G.num_nodes
print N
print G.num_edges


# eidxls=G.edges_()
# print eidxls
with open('lines.csv','rb') as csvfile:
	q=csv.DictReader(csvfile)
	for row in q:
		id,Resistance,reactance,bus0,bus1,voltage=(row['name'],row['r_ohmkm'],row['x_ohmkm'],row['bus0'],row['bus1'],row['voltage'])
		bus0=bus0.split('_')
		bus1=bus1.split('_')
		Impedance=Resistance+'+j'+reactance
		if int(voltage)>220001:
			volt_type='Industrial'
		else:
			volt_type='Domestic'
		if G.has_edge(bus0[0],bus1[0]):
			 w=G.weight(bus0[0],bus1[0])+1
			 # print i,j,w
			 G.set_weight(bus0[0],bus1[0],w)
			 G.set_edge_data(bus0[0],bus1[0],data=[volt_type,Impedance])
			 continue
		G.add_edge(bus0[0],bus1[0],data=[volt_type,Impedance])
		#G.set_edge_data(bus0[0],bus1[0],data=Impedance)
with open('vertices_de_power_151109.csv','rb') as csvfile:
	q=csv.DictReader(csvfile)
	for row in q:
		n_data=[]
		v_id,lon,lat,node_type=(row['v_id'],row['lon'],row['lat'],row['typ'])
		gps = lat+','+lon
		if G.__contains__(v_id):
			b=G.node_data(v_id)
			n_data.append(b)
			n_data.append(gps)
			n_data.append(node_type)
			G.set_node_data(v_id,n_data)
			continue
		#id=int(id)
print G.num_edges
print G.num_nodes
G.compact()
zen.io.gml.write(G,'Project_network. gml',write_data=(True),use_zen_data=(True))
