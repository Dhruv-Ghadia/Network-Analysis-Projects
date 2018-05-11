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
import random



def modularity(G,c):
	d = dict()
	for k,v in c.iteritems():
		for n in v:
			d[n] = k
	Q, Qmax = 0,1
	for u in G.nodes_iter():
		for v in G.nodes_iter():
			if d[str(u)] == d[str(v)]:
				Q += ( int(G.has_edge(v,u)) - G.degree(u)*G.degree(v)/float(G.num_edges) )/float(G.num_edges)
				Qmax -= ( G.degree(u)*G.degree(v)/float(G.num_edges) )/float(G.num_edges)
	return Q, Qmax

def RGBToHTMLColor(rgb_tuple,X,Y):
	""" convert an (R, G, B) tuple to #RRGGBB """
	hexcolor = '#%02x%02x%02x' % rgb_tuple
	# that's it! '%02x' means zero-padded, 2-digit hex values
	return hexcolor,X,Y

def HTMLColorToRGB(colorstring):
	""" convert #RRGGBB to an (R, G, B) tuple """
	colorstring = colorstring.strip()
	if colorstring[0] == '#': colorstring = colorstring[1:]
	if len(colorstring) != 6:
		raise ValueError, "input #%s is not in #RRGGBB format" % colorstring
	r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
	r, g, b = [int(n, 16) for n in (r, g, b)]
	return (r, g, b)

def color_interp(color1,color2,color3,color4,v,nidx,m=0,m1=1,m2=2,m3=3,m4=4):
	c1 = array(HTMLColorToRGB(color1))
	c2 = array(HTMLColorToRGB(color2))
	c3 = array(HTMLColorToRGB(color3))
	c4 = array(HTMLColorToRGB(color4))
	if v == m:
		c = tuple(c3)
		x = 400 + 330 * math.cos(math.radians(360.0*nidx/G.num_nodes))
		y = 300 + 330 * math.sin(math.radians(360.0*nidx/G.num_nodes))
		# y=random.randint(0,600)
		# x=random.randint(400,800)
	elif v ==m1:
		c = tuple(c2)
		x = 400 + 230 * math.cos(math.radians(360.0*nidx/G.num_nodes))
		y = 300 + 230 * math.sin(math.radians(360.0*nidx/G.num_nodes))
		# y=random.randint(0,300)
		# x=random.randint(0,399)
	elif v == m2:
		c = tuple(c1)
		x = 400 + 170 * math.cos(math.radians(360.0*nidx/G.num_nodes))
		y = 300 + 170 * math.sin(math.radians(360.0*nidx/G.num_nodes))
		# y=random.randint(301,600)
		# x=random.randint(0,399)
	elif v==m3:
		c= tuple(c2)
		x = 400 + 230 * math.cos(math.radians(360.0*nidx/G.num_nodes))
		y = 300 + 230 * math.sin(math.radians(360.0*nidx/G.num_nodes))
		# y=random.randint(0,300)
		# x=random.randint(0,399)
	elif v==m4:
		c= tuple(c3)
		# x=200
		# y=600
		x = 400 + 330 * math.cos(math.radians(360.0*nidx/G.num_nodes))
		y = 300 + 330 * math.sin(math.radians(360.0*nidx/G.num_nodes))
		# y=random.randint(0,600)
		# x=random.randint(400,800)
	else:
		#c = tuple( c1 + (c2-c1)/(M-m)*(v-m) ) # linear interpolation of color
		c = tuple( c1 + (c2-c1)*(1 - exp(-2*(v-m)/(m1-m))) ) # logistic interpolation of color
	return RGBToHTMLColor(c,x,y)

def color_interp1(color1,color2,color3,color4,v,nidx,m=0,m1=1,m2=2,m3=3,m4=4):
	c1 = array(HTMLColorToRGB(color1))
	c2 = array(HTMLColorToRGB(color2))
	c3 = array(HTMLColorToRGB(color3))
	c4 = array(HTMLColorToRGB(color4))
	if v == m:
		c = tuple(c1)
		x = 400 + 150 * math.cos(math.radians(360.0*(nidx+2)/G.num_nodes))
		y = 300 + 150 * math.sin(math.radians(360.0*(nidx+2)/G.num_nodes))
		# y=random.randint(0,600)
		# x=random.randint(400,800)
	elif v ==m1:
		c = tuple(c2)
		x = 400 + 330 * math.cos(math.radians(360.0*(nidx+2)/G.num_nodes))
		y = 300 + 330 * math.sin(math.radians(360.0*(nidx+2)/G.num_nodes))
		# y=random.randint(0,300)
		# x=random.randint(0,399)
	elif v == m2:
		c = tuple(c3)
		x = 400 + 100 * math.cos(math.radians(360.0*(nidx+2)/G.num_nodes))
		y = 300 + 100 * math.sin(math.radians(360.0*(nidx+2)/G.num_nodes))
		# y=random.randint(301,600)
		# x=random.randint(0,399)
	elif v==m3:
		c= tuple(c4)
		x = 400 + 230 * math.cos(math.radians(360.0*(nidx+2)/G.num_nodes))
		y = 300 + 230 * math.sin(math.radians(360.0*(nidx+2)/G.num_nodes))
		# y=random.randint(0,300)
		# x=random.randint(0,399)
	elif v==m4:
		c= tuple(c3)
		# x=200
		# y=600
		x = 400 + 330 * math.cos(math.radians(360.0*(nidx+2)/G.num_nodes))
		y = 300 + 330 * math.sin(math.radians(360.0*(nidx+2)/G.num_nodes))
		# y=random.randint(0,600)
		# x=random.randint(400,800)
	else:
		#c = tuple( c1 + (c2-c1)/(M-m)*(v-m) ) # linear interpolation of color
		c = tuple( c1 + (c2-c1)*(1 - exp(-2*(v-m)/(m1-m))) ) # logistic interpolation of color
	return RGBToHTMLColor(c,x,y)

def color_by_value(d3,G,x,color1='#9900cc',color2='#009900',color3='#FF0000',color4='#AF3214'):
	d3.set_interactive(False)
	m = min(x)
	M = max(x)
	for i in G.nodes_():
		r,X,Y = color_interp(color1,color2,color3,color4,x[i-1],i)
		d3.stylize_node_(i, d3js.node_style(fill=r))
		# d3.update()
		d3.position_node_(i,X,Y)
	d3.update()
	d3.set_interactive(False)

def color_by_value1(d3,G,x,color1='#9900cc',color2='#009900',color3='#FF0000',color4='#FFCC00'):
	d3.set_interactive(False)
	m = min(x)
	M = max(x)
	for i in G.nodes_():
		r,X,Y = color_interp1(color1,color2,color3,color4,x[i-1],i)
		d3.stylize_node_(i, d3js.node_style(fill=r))
		# d3.update()
		d3.position_node_(i,X,Y)
	d3.update()
	d3.set_interactive(False)

#==========================================================================================================
G = zen.io.gml.read('Project_network. gml',weighted=True)
renewable = ['Run of River','Storage Hydro','Geothermal','Solar','Wind Offshore','Wind Onshore']
non_renewable = ['Waste','Nuclear','Gas','Hard Coal','Brown Coal','oil']
station_type = ['plant','substation', 'generator','auxillary_T_node']
source_type = zeros(G.num_nodes)
gps_data = []
n_type = []
red=0
green=0
purple=0
#gps_data = zeros(G.num_nodes)
#n_type = zeros(G.num_nodes)
#color code on source
for i in G.nodes():
	# print i
	source = G.node_data(i)
	source = source['zenData']
	source_data = source[0]
	gps_data.append(source[1])
	n_type.append(source[2])
#	gps_data[i] = source[1]
#	n_type[i] = source[2]
	source_data = source_data.split(', ')
	com_ren = set(source_data).intersection(renewable)
#	print 'source and renewable intersection:',com_ren
	com_non_ren = set(source_data).intersection(non_renewable)
#	print 'source and non renewable intersection:',com_non_ren
	j = int(i)-1
	if len(com_ren) == len(source):
		source_type[j] = 1
		green=green+1
	elif len(com_non_ren) == len(source):
		source_type[j] = 0
		red=red+1
	elif len(com_ren) == len(com_non_ren):
		source_type[j] = 2
		purple=purple+1
	elif len(com_ren) > 6 -len(com_ren):
		source_type[j] = 3
		green=green+1
	else:
		source_type[j] = 4
		red=red+1
# print 'green %i' %green
# print 'red %i' %red
# print 'Black %i' %purple
for i,d in enumerate(gps_data):
	d_split = d.split(',')
	gps_data[i] = [float(loc)for loc in d_split]
# plotting on map
plant_idx = []
substation_idx = []
generator_idx = []
tnode_idx = []

# modularity groups formation
for i,s in enumerate(n_type):
	if s == "plant":
		plant_idx.append(str(i+1))
	elif s == "substation":
		substation_idx.append(str(i+1))
	elif s == "generator":
		generator_idx.append(str(i+1))
	elif s == "auxillary_T_node":
		tnode_idx.append(str(i+1))
	else:
		continue

production = {"plant":plant_idx,
			  "generator":generator_idx,
			  "substation":substation_idx,
			  "tnode":tnode_idx}

Q, Qmax = modularity(G,production)
print Q, Qmax
print 'Modularity: %1.4f / %1.4f' % (numpy.abs(Q),numpy.abs(Qmax))

domestic = 0
industrial = 0
plant_u_no = 0
plant_v_no=0
for u,v,data in G.edges_iter(data=True):
	if data[0] == 'Domestic':
		domestic = domestic+1
		data_u = G.node_data(u)
		data_u = data_u['zenData']
		data_u = data_u[2]
		data_v = G.node_data(v)
		data_v = data_v['zenData']
		data_v = data_v[2]
		if data_u == 'plant':
			plant_u_no= plant_u_no + 1
		if data_v == 'plant':
			plant_v_no = plant_v_no + 1

	elif data[0] == 'Industrial':
		industrial = industrial +1
		data_u1 = G.node_data(u)
		data_u1 = data_u1['zenData']
		data_u1 = data_u1[2]
		data_v1 = G.node_data(v)
		data_v1 = data_v1['zenData']
		data_v1 = data_v1[2]
		if data_u == 'plant':
			plant_u_no= plant_u_no + 1
		if data_v == 'plant':
			plant_v_no = plant_v_no + 1
total = domestic + industrial
print domestic,industrial
print 'Supply % based on the voltage available:'
print 'domestic supply lines is %i and is about %1.4f' %(domestic,domestic*100/float(total))
print 'industrial supply lines is %i and is about %1.4f' %(industrial,industrial*100/float(total))

total = green+purple+red
print 'green i.e. renewable energy is %f' %(float(green)/float(total))
print 'red i.e. non_renewable energy is %1.2f' %(float(red)/float(total))
print 'purple i.e. combined energy is %1.2f' %(float(purple)/float(total))

#G = zen.io.gml.read('Project_network. gml',weighted=True)
d3 = d3js.D3jsRenderer(G, event_delay=0.1, interactive=False, autolaunch=False)
d3.update()
sleep(1)

#color coding:
renewable = ['Run of River','Storage Hydro','Geothermal','Solar','Wind Offshore','Wind Onshore']
non_renewable = ['Waste','Nuclear','Gas','Hard Coal','Brown Coal','oil']
station_type = ['plant','substation', 'generator','auxillary_T_node']
source_type = zeros(G.num_nodes)

print 'green %i' %green
print 'red %i' %red
print 'Black %i' %purple

#color code on type
nt = zeros(G.num_nodes)
for i,n in enumerate(n_type):
	if n == station_type[0]:
		nt[i] = 0
	elif n == station_type[1]:
		nt[i] = 1
	elif n == station_type[2]:
		nt[i] = 2
	elif n == station_type[3]:
		nt[i] = 3
	else:
		nt[i] = 4

d3.set_title('Highlighting Nodes based on source...')

print 'updating color coding based on source'
color_by_value(d3,G,source_type)
d3.update()
sleep(1)

d3.set_title('Highlighting Nodes based on Node type...')

print 'updating color coding based on node type'
color_by_value1(d3,G,nt)
d3.update()
sleep(1)

from gmplot import gmplot
# Place map
gmap = gmplot.GoogleMapPlotter(52.52437, 13.41053, 13)
gmap.scatter(gps_data[0], gps_data[1], '#3B0B39', size=40, marker=False)
# Draw
gmap.draw("germany.html")
d3.stop_server()