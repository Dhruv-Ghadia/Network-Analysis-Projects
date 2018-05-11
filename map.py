# Place map
import zen
import sys
import numpy
from numpy import *
import numpy.linalg as la
import matplotlib.pyplot as plt
sys.path.append('../zend3js/')
import d3js
import string
import time
from time import sleep
import csv
import gmplot
html_color_codes =[
'aliceblue',
'antiquewhite',
'aqua',
'aquamarine',
'azure',
'beige',
'bisque',
'black',
'blanchedalmond',
'blue',
'blueviolet',
'brown',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
'cornsilk',
'crimson',
'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgray',
'darkgreen',
'darkkhaki',
'darkmagenta',
'darkolivegreen',
'darkorange',
'darkorchid',
'darkred',
'darksalmon',
'darkseagreen',
'darkslateblue',
'darkslategray',
'darkturquoise',
'darkviolet',
'deeppink',
'deepskyblue',
'dimgray',
'dodgerblue',
'firebrick',
'floralwhite',
'forestgreen',
'fuchsia',
'gainsboro',
'ghostwhite',
'gold',
'goldenrod',
'gray',
'green',
'greenyellow',
'honeydew',
'hotpink',
'indianred',
'indigo',
'ivory',
'khaki',
'lavender',
'lavenderblush',
'lawngreen',
'lemonchiffon',
'lightblue',
'lightcoral',
'lightcyan',
'lightgoldenrodyellow',
'lightgray',
'lightgreen',
'lightpink',
'lightsalmon',
'lightseagreen',
'lightskyblue',
'lightslategray',
'lightsteelblue',
'lightyellow',
'lime',
'limegreen',
'linen',
'magenta',
'maroon',
'mediumaquamarine',
'mediumblue',
'mediumorchid',
'mediumpurple',
'mediumseagreen',
'mediumslateblue',
'mediumspringgreen',
'mediumturquoise',
'mediumvioletred',
'midnightblue',
'mintcream',
'mistyrose',
'moccasin',
'navajowhite',
'navy',
'oldlace',
'olive',
'olivedrab',
'orange',
'orangered',
'orchid',
'palegoldenrod',
'palegreen',
'paleturquoise',
'palevioletred',
'papayawhip',
'peachpuff',
'peru',
'pink',
'plum',
'powderblue',
'purple',
'red',
'rosybrown',
'royalblue',
'saddlebrown',
'salmon',
'sandybrown',
'seagreen',
'seashell',
'sienna',
'silver',
'skyblue',
'slateblue',
'slategray',
'snow',
'springgreen',
'steelblue',
'tan',
'teal',
'thistle',
'tomato',
'turquoise',
'violet',
'wheat',
'white',
'whitesmoke',
'yellow',
'yellowgreen'
]
G = zen.io.gml.read('Project_network. gml',weighted=True)
gmap = gmplot.GoogleMapPlotter(50.9168399490798, 9.25938382024592, 13)
location =[]
lat =[]
lon =[]
for i in G.nodes():
	data = G.node_data(i)
	data = data['zenData']
	gps = data[1]
	loc = gps.split(',')
	loc_tup = (loc[0],loc[1])
#	print loc_tup
	location.append(loc_tup)
#	gps1 = '(' + gps + ')'
#	location.append(gps1)
	lat.append(loc[0])
	lon.append(loc[1])
#====================================================================================
## Marker for nodes
#for k in range(0,len(lat)):
#	hidden_gem_lat, hidden_gem_lon = float(lat[k]),float(lon[k])
##	gmap.scatter(hidden_gem_lat, hidden_gem_lon, '#FF08FF', size=50, marker=True)
#	gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')
#gmap.draw("Nodes.html")
#====================================================================================
gmap = gmplot.GoogleMapPlotter(float(lat[0]),float(lon[0]), 13)
for k in range(0,len(lat)):
	hidden_gem_lat, hidden_gem_lon = float(lat[k]),float(lon[k])
	data = G.node_data(k+1)
	data = data['zenData']
#	print data
	print data[2]
	if data[2] == "plant":
		gmap.marker(hidden_gem_lat, hidden_gem_lon, 'azure')
	elif data[2] == "substation":
		gmap.marker(hidden_gem_lat, hidden_gem_lon, 'lavender')
	elif data[2] == "generator":
		gmap.marker(hidden_gem_lat, hidden_gem_lon, 'orange')
	else:
		gmap.marker(hidden_gem_lat, hidden_gem_lon, 'lime')
#gmap.draw("Nodes.html")
#gmap.draw("colornodes.html")
color =numpy.random.choice(html_color_codes)
comm1=[]
comm2=[]
for i in G.nodes():
    u=G.node_data(i)
    u=u["zenData"]
    t=u[1]
    x,y=t.split(',')
    x=float(x)
    y=float(y)
    comm1.append(x)
    comm2.append(y)
gmap.plot(comm1, comm2, 'ivory', edge_width=1)

gmap.draw("map.html")
