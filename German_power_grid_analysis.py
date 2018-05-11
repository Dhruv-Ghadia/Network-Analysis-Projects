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
from gmplot import gmplot
from geopy.geocoders import Nominatim
geolocator = Nominatim()

#==========================================================================================================
#================== Fuction Definitions and Hekper functons================================================
def propagate(G,d3,x,steps,slp=0.5,keep_highlights=False,update_at_end=False):
	interactive = d3.interactive
	d3.set_interactive(False)
	A = G.matrix().T  # adjacency matrix of the network G
	d3.highlight_nodes_(list(where(x>0)[0]))
	d3.update()
	sleep(slp)
	cum_highlighted = sign(x)
	for i in range(steps): # the brains
		x = sign(dot(A,x)) # the brains
		cum_highlighted = sign(cum_highlighted+x)
		if not update_at_end:
			if not keep_highlights:
				d3.clear_highlights()
			d3.highlight_nodes_(list(where(x>0)[0]))
			d3.update()
			sleep(slp)
	if update_at_end:
		if not keep_highlights:
			d3.clear_highlights()
			d3.highlight_nodes_(list(where(x>0)[0]))
		else:
			d3.highlight_nodes_(list(where(cum_highlighted>0)[0]))
		d3.update()
	d3.set_interactive(interactive)
	if keep_highlights:
		return cum_highlighted
	else:
		return x


# prints the top five (num) nodes according to the centrality vector v
# v takes the form: v[nidx] is the centrality of node with index nidx
def print_top(G,v, num=5, color='cornflowerblue'):
	idx_list = [(i,v[i]) for i in range(len(v))]
	idx_list = sorted(idx_list, key = lambda x: x[1], reverse=True)
	for i in range(min(num,len(idx_list))):
		nidx, score = idx_list[i]
		print '  %i. %s (%1.4f)' % (i+1,G.node_object(nidx),score)
		print 'Type of Fuel source used for power Generation:'
		node_data = G.node_data_(nidx)
		data = node_data['zenData']
		fuel_source = data[0]
		print fuel_source
		print 'The type of node is %s'%data[2]
#		print '%s' %G.node_data_(nidx)
		gps = data[1]
		loc = gps.split(',')
#		loc_tup = (loc[0],loc[1])
		lat, lon = float(loc[0]),float(loc[1])
		gmap.marker(lat, lon, color)
#		location = geolocator.reverse(gps) # To get the address file
#		print 'location address is:'
#		print(location.address)
	gmap.draw("print_top.html")
		#print '  %i. %s' % (i+1,G.node_object(idx))

# returns the index of the maximum of the array
# if two or more indices have the same max value, the first index is returned
def index_of_max(v):
	return numpy.where(v == max(v))[0]

#CONFIG MODELS
def degree_sequence(G):
	return [degree for degree,freq in enumerate(zen.degree.ddist(G,normalize=False)) for f in range(int(freq))]

def configuration_model(degree_sequence,G1=None):
	import numpy.random as numpyrandom
	if G1 is None:
		G1 = zen.Graph()

	n = len(degree_sequence)
	for i in range(n):
		G1.add_node(i)

	# this is twice the number of edges, needs to be even
	assert mod(sum(degree_sequence),2) == 0, 'The number of edges needs to be even; the sum of degrees is not even.'
	num_edges = sum(degree_sequence)/2

	# the number of edges should be even
	assert mod(num_edges,2) == 0, 'The number of edges needs to be even.'

	stubs = [nidx for nidx,degree in enumerate(degree_sequence) for d in range(degree)]
	stub_pairs = numpyrandom.permutation(num_edges*2)

	self_edges = 0
	multi_edges = 0
	for i in range(num_edges):
		uidx = stubs[stub_pairs[2*i]]
		vidx = stubs[stub_pairs[2*i+1]]
		if uidx == vidx:
			self_edges += 1
		if G1.has_edge_(uidx,vidx):
			eidx = G1.edge_idx_(uidx,vidx)
			G1.set_weight_(eidx, G1.weight_(eidx)+1 )
			multi_edges += 1
		else:
			G1.add_edge_(uidx,vidx)

	print 'self edges: %i,  multi-edges: %i' % (self_edges,multi_edges)
	return G1

def Clustering(G):
	O=G.nodes()
	n=G.num_nodes
	v=[]
	for i in O:
		u=G.degree(i)
		v.append(u)
	k=numpy.square(v)
	k2=float(numpy.sum(k))/n
	k1=float(numpy.sum(v))/n
	D=float(numpy.square(k2-k1))/(k1*k1*k1)
	return D/n
def Stats(G):
	o=G.num_nodes
	p=G.num_edges
	C=2*p/o
	P=float(C)/(o-1)
	D=Clustering(G)
	J=degree_sequence(G)
	return o,p,C,P,D,J

##############################################################
def cocitation(G):
    G1=zen.Graph()
    g=G.nodes()
    H=0
    H1=0
    e=0
    j=0
    y=0
    for i in range(0,len(g)):
        G1.add_node(g[i])
    for j in range(0,len(g)):
        u=G.neighbors(g[j])
        r=len(u)
        if r==0:
            r=r+1
        for y in range(0,len(g)):
            w=[]
            v=G.neighbors(g[y])
            p=len(v)
            if p==0:
                p=p+1
            for s in range(0,r):
                if len(u)==0:
                    continue
                H=G.weight(u[s],g[j])
                for e in range(0,p):
                    if len(v)==0:
                        continue
                    H1=G.weight(v[e],g[y])
                    if len(u)==0:
                        continue
                    if len(v)==0:
                        continue
                    if (v[e]==u[s]):
                        w.append(H*H1)
                        count=numpy.sum(w)
                        if G1.has_edge(g[j],g[y])==True:
                            G1.set_weight(g[j],g[y],count)
                            continue
                        if g[j]==g[y]:
                            continue
                        G1.add_edge(g[j],g[y],weight=count)
    #X=G1.matrix()
    #numpy.fill_diagonal(X,0)
    #print X.sum().sum()
    return G1

def RGBToHTMLColor(rgb_tuple):
	""" convert an (R, G, B) tuple to #RRGGBB """
	hexcolor = '#%02x%02x%02x' % rgb_tuple
	# that's it! '%02x' means zero-padded, 2-digit hex values
	return hexcolor

def HTMLColorToRGB(colorstring):
	""" convert #RRGGBB to an (R, G, B) tuple """
	colorstring = colorstring.strip()
	if colorstring[0] == '#': colorstring = colorstring[1:]
	if len(colorstring) != 6:
		raise ValueError, "input #%s is not in #RRGGBB format" % colorstring
	r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
	r, g, b = [int(n, 16) for n in (r, g, b)]
	return (r, g, b)

def color_interp(color1,color2,v,m=0,M=1):
	c1 = array(HTMLColorToRGB(color1))
	c2 = array(HTMLColorToRGB(color2))
	if v > M:
		c = tuple(c2)
	elif v < m:
		c = tuple(c1)
	else:
		#c = tuple( c1 + (c2-c1)/(M-m)*(v-m) ) # linear interpolation of color
		c = tuple( c1 + (c2-c1)*(1 - exp(-2*(v-m)/(M-m))) ) # logistic interpolation of color
	return RGBToHTMLColor(c)

def color_by_value(d3,G,x,color1='#77BEF5',color2='#F57878'):
	d3.set_interactive(False)
	m = min(x)
	M = max(x)
	for i in G.nodes_iter_():
		d3.stylize_node_(i, d3js.node_style(fill=color_interp(color1,color2,x[i])))
	d3.update()
	d3.set_interactive(True)

# calculate the alpha and sigma value using newman 8.6 & 8.7
def cal_alpha_sigma(G, kmin):
    summation = 0
    alpha = 0
    G_nodes = G.num_nodes
    deg = numpy.zeros((G_nodes))
    for i in range(0, G_nodes):
        deg[i] = G.degree_(i)
    deg_seq = numpy.sort(deg)
    N = sum(deg >= kmin)
    print '\nNumber of nodes(N) with degree >= kmin is %i' %N
    den = kmin - 0.5
    for i in range(0,G_nodes):
        if (deg_seq[i] >= kmin):
            num = deg_seq[i]
            summation = summation + numpy.log(num/den)
    alpha = 1 + N / summation
    sigma = (alpha - 1)/ numpy.sqrt(N)
    return alpha, sigma

## Plots the degree distribution and calculates the power law coefficent
def calc_powerlaw(G,kmin):
	ddist = zen.degree.ddist(G,normalize=False)
	cdist = zen.degree.cddist(G,inverse=True)
	k = numpy.arange(len(ddist))

	plt.figure(figsize=(12,8))
	plt.subplot(121)
	plt.bar(k,ddist, width=0.8, bottom=0, color='b')
	plt.xlabel('k')
	plt.ylabel('p(k)')
	plt.title('Degree distribution')
	plt.subplot(122)
	plt.loglog(k,cdist)
	plt.xlabel('log(k)')
	plt.ylabel('log(cdf)')
	plt.title('Cummulative distribution')

	alpha, sigma = cal_alpha_sigma(G, kmin) # calculate using Newman (8.6)!
	#sigma = 0 # calculate using Newman (8.7)!
	print '%1.2f +/- %1.2f' % (alpha,sigma)

	plt.show()


def check_friendship_paradox(G):
    G_nodes = G.num_nodes
    k_sum = 0
    k_sqr_sum = 0
    for i in range(0, G_nodes):
        i_deg = G.degree_(i)
        k_sum = k_sum + i_deg
        k_sqr_sum = k_sqr_sum + i_deg * i_deg
    k = k_sum / G_nodes
    k_sqr = k_sqr_sum / G_nodes
    print '\n<k^2> = %i and <k> = %i' %(k_sqr, k)
    if (float(k_sqr)/float(k) > float(k)):
        print'\nNeighbour of a node has more neighbors than the node -> friendship paradox is observed'
    else:
        print '\nFriendship paradox is not observed'
#=======================================================================================
#======================= Network Analysis ==============================================
#=======================================================================================

G = zen.io.gml.read('Project_network. gml',weighted=True)
A = G.matrix()
N = G.num_nodes
gmap = gmplot.GoogleMapPlotter(51.0603668474477,9.25938382024592, 13)
# d3 = d3js.D3jsRenderer(G, event_delay=0.1, interactive=False, autolaunch=False)
# d3.update()
# sleep(1)
# Degree Centrality
print '\nDegree Centrality:'
v1 = numpy.zeros((N,1))

for i in range(0,N): #loops takes the total number of nodes in the entire network
	a = G.neighbors_(i) #takes the neighbors of each node and puts them in an array
	N1 = len(a)
	wt = 0
	for j in range(0,N1): #loops through the neighbors of a particular node
		w2 = a[j]
		wt = wt + G.weight(G.node_object(a[j]),G.node_object(i))
	v1[i] = wt

print '\nThe top five characters based on degree centrality are:'
print_top(G,v1, num=5,color='orange')

print '\n============================================='
# Eigenvector Centrality
print '\nEigenvector Centrality (by Zen):'
G1 = zen.algorithms.centrality.eigenvector_centrality_(G,weighted=True)
print_top(G,G1, num=5,color='green')

print '\n============================================='
# PageRank
print '\nPageRank'
D = numpy.zeros((N,N))
vec_1 = numpy.ones((N,1))
beta = 1
for i in range(0, N):
	node_i_out_degree = G.degree_(i)
	if node_i_out_degree < 1:
		node_i_out_degree = 1
	D[i][i] = node_i_out_degree
alpha = 0.85
AD_inv = numpy.dot(A, la.inv(D))
diff = numpy.eye(N) - numpy.dot(alpha, AD_inv)
page_rank_centrality = numpy.dot(la.inv(diff), vec_1)
print '\n Page Rank Centrality with alpha = %1.2f, beta = %i :' % (alpha, beta)
print_top(G,page_rank_centrality,num=5,color='pink')


print '\n============================================='
# Betweenness Centrality
print '\nBetweenness Centrality'
G4 = zen.algorithms.centrality.betweenness_centrality_(G,weighted = True)
print_top(G,G4, num = 5, color='red')

###############################################################################
X=zen.components(G)
print 'Number of components %s'%len(X)
###############################################################################
A=zen.algorithms.clustering.gcc(G)
print 'Global Clustring= %s'%A


G1=cocitation(G)
#print G1.matrix()

###############################################################################
print 'Network Stats'
H=degree_sequence(G)
# G=zen.Graph()
# G=configuration_model(H)
o,p,C,P,D,J=Stats(G)
print "Network Number of Nodes %s" %o
print "Network Number of edges %s" %p
print "Network Average Degree %s"%C
print "Network Clustering coefficent %s" %D
print "Network Global Clustering %s"%zen.clustering.gcc(G)

print 'power law'
calc_powerlaw(G, 10)
###############################################################################
cset = zen.algorithms.community.louvain(G)
h=open('community.txt','w')
k=[]
for i in cset:
	k.append(len(i))
	for j in i:
		j=str(j)
		h.write(j)
		h.write(', ')
	h.write('\n')
h.close()
print 'Number of communities %i'%len(cset)
plt.figure()
plt.hist(k,20)
plt.xlabel('community sizes')
plt.show()
# h=open('community.txt','r')
# for line in h:
# 	k=line.replace(',','')
# 	K=k.split()
# 	for i in K:
# 		print G.node_data(int(i))

#=============================================================================================
print '\nFriendship paradox\n'
check_friendship_paradox(G)

###############################################################################
G3=G.copy()
print 'CONFIG MODEL**********************************************'
H=degree_sequence(G3)
G3=zen.Graph()
G3=configuration_model(H,G3)
o,p,C,P,D,J=Stats(G3)
print "Config MODEL Number of Nodes %s" %o
print "Config MODEL Number of edges %s" %p
print "Config MODEL Average Degree %s"%C
d= zen.diameter(G3)
print 'Network has a diameter of %i.' % d
# for i in X:
# 	for j in Y:
# 		numpy.delete(D,i)
###############################################################################
#####################################################################
print 'Percolation*************************************************'
G2=G.copy()
X=G2.edges_()
p=arange(0,1,0.2)
comp_list=[]
for j in p:
	for i in X:
		if random.random() <= j:
			if i in G2.edges_():
				G2.rm_edge_(i)
	y=zen.components(G2)
	comp_list.append(len(max(y)))
plt.figure()
plt.plot(p,comp_list,'b',linewidth=3)
plt.title('Percolation Vs Number of Components')
plt.xlabel('Occupation probabilty')
plt.ylabel('Size of largest Component')
plt.show()
print 'stats after percolation'
print 'Number of Edges after percolation = %s'%G2.num_edges
y=zen.components(G2)
print 'Number of components after percolation = %s' %len(y)
d= zen.diameter(G2)
print 'Diameter after percolation %i.' % d

print 'Attack*******************************************************'
#deg_seq=degree_sequence(G)
G4=G.copy()
node_list=G4.nodes()
p=arange(0,495)
p=arange(0,10,1)
comp_list=[]
for j in p:
	G4=G.copy()
	for i in node_list:
		if i in G4.nodes():
			if G4.degree(i)>=j:
				G4.rm_node(i)
	y=zen.components(G4)
	if len(y) == 0:
#		print max(y)
		comp_list.append(0)
	else:
		comp_list.append(len(max(y)))
#comp_list.push(0)
print G4.num_nodes
#p = p/G.num_nodes
plt.figure()
plt.plot(p,comp_list,'b',linewidth=3)
plt.title('Attack Vs Number of Components')
plt.xlabel('Degree of Nodes attacked')
plt.ylabel('Number of components')
plt.show()
print 'Number of Edges after Attack on high degree nodes = %s'%G4.num_edges
y=zen.components(G4)
print 'Number of components after Attack = %s' %len(y)

###############################################################################
print 'SIR + Diffusion ********************************************************'

A=G.matrix()
#d3 = d3js.D3jsRenderer(G, event_delay=0.1, interactive=False, autolaunch=False)
#d3.update()
#sleep(1)

dt = 0.05 # the "infintesimal" size steps we take to integrate
T = 6 # the end of the simulation time
time = linspace(0,T,int(T/dt)) # the array of time points spaced by dt

## DIFFUSION ==============================================
print '============================\nDIFFUSION\n'
x = zeros(G.num_nodes) # the state vector
x[1] = 1
#color_by_value(d3,G,x) # this colors the network according to the value of x
G.compact()
#G.validate()
A = G.matrix()
I = numpy.identity(G.num_nodes)
#D = dot(I,sum(A,axis =1))
D = I * numpy.sum(A,axis =1)
#L = D - A
L = A - D
diff_Con = 1
k, v = la.eig(L)
eig_idx = numpy.where(k == k.min())
eig_vec = abs(v[:,eig_idx])
#eig_idx = numpy.where(k == k.max())
#eig_vec = abs(v[:,eig_idx])
norm_eig_vec = la.norm(eig_vec)
n_eig_vec = eig_vec/norm_eig_vec
print 'simulating diffusion...'

error = numpy.zeros(len(time))
for i,t in enumerate(time):
    # at each time point update the value of x
    x = x + numpy.dot(diff_Con, numpy.dot(L, x)) * dt
    normal_x = x
    x = x / numpy.linalg.norm(x)
    error[i] = numpy.sqrt(numpy.sum((x-n_eig_vec)**2))
    x = x * numpy.linalg.norm(x)
#    color_by_value(d3,G,normal_x)
#    sleep(0.1)


plt.figure()
plt.plot(time,error) # replace xvalue and yvalue with what you want to plot
plt.xlabel('t')
plt.ylabel('Eucilidean distance') # change this label
plt.title('Diffusion model')
plt.show()

###############################################################################
## SI MODEL ===============================================
print '============================\nSI MODEL\n'
x = zeros(G.num_nodes) # the state vector
x[16] = 1
bt=1
s=1-x
var_x=zeros(len(time))
var_s=zeros(len(time))
for i,t in enumerate(time):
	# at each time point update the value of
	f3=-bt*s*numpy.dot(A,x)
	s=s+f3*dt
	Y=bt*s*numpy.dot(A,x)
	x=x+Y*dt
	var_x[i]=numpy.mean(x)
	var_s[i]=numpy.mean(s)
plt.figure()
plt.plot(time,var_x,label ='infection')# replace xvalue and yvalue with what you want to plot
plt.plot(time,var_s,'r',label ='susceptible')
plt.xlabel('time')
plt.ylabel('fraction of infected nodes') # change this label
plt.legend()
plt.title('SI model')
plt.show()

###############################################################################
## SIR MODEL ==============================================
print '============================\nSIR MODEL\n'
x = zeros(G.num_nodes)
r = zeros(G.num_nodes) # the state vector
u,v=la.eig(A)
lamb=u.max()
#for i in range(0,len(hubs)):
#	x[hubs[i]] = 1
x[55] = 1
bt=1
g1=1
s=1-x
var_x=zeros(len(time))
var_s=zeros(len(time))
var_r=zeros(len(time))
for i,t in enumerate(time):
	# at each time point update the value of
	f3=-bt*s*numpy.dot(A,x)
	f4=g1*x
	s=s+f3*dt
	r=r+f4*dt
	Y=bt*s*numpy.dot(A,x)-g1*x
	x=x+Y*dt
	var_x[i]=numpy.mean(x)
	var_s[i]=numpy.mean(s)
	var_r[i]=numpy.mean(r)
plt.figure()
plt.plot(time,var_x,label = 'infection')# replace xvalue and yvalue with what you want to plot
plt.plot(time,var_s,'r', label = 'susceptible')
plt.plot(time,var_r,'g',label = 'recovery')
plt.xlabel('time')
plt.ylabel('fraction of infected nodes') # change this label
plt.legend()
plt.title('SIR model')
plt.show()

###############################################################################
print 'Hubs are as follows:'
gmap = gmplot.GoogleMapPlotter(52.52437, 13.41053, 13)
G = zen.io.gml.read('Project_network. gml',weighted=True)
count = 0
hubs = []
for i in G.nodes():
	if G.degree(i) >=10:
		count = count +1

		hubs.append(i)
		data = G.node_data(i)
		data = data['zenData']
#	print data
		print '%d) Node no: %d and it is a %s' %(count,i,data[2])
#		print data[2]
		gps = data[1]
		loc = gps.split(',')
#		loc_tup = (loc[0],loc[1])
		lat, lon = float(loc[0]),float(loc[1])
		gmap.marker(lat, lon, 'green')
#		location = geolocator.reverse(gps)
#		print 'location address of the hub is:'
#		print(location.address)

gmap.draw('hubs.html')
