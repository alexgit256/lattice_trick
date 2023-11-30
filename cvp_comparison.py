"""
To be launched as:
sage cvp_comparison.py [1> out.txt] [--plot]
"""

from sage.all_cmdline import *   # import sage library

from time import perf_counter
import os, re, sys
from pathlib import Path

import fpylll
from fpylll import IntegerMatrix, CVP
from fpylll.fplll.gso import MatGSO
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ, FPLLL, Enumeration, CVP, SVP
from fpylll.fplll.enumeration import EvaluatorStrategy

import time, pickle
from math import sqrt
import numpy as np
from itertools import islice #to get first n elems of generator

#for plots
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches

from lattices import generate_short_vectors
RR = RealField(100)

def enum(B,t,r, count = 2000):
    out = []
    b0, b1 = vector(B[0]), vector(B[1])
    mu = RR(b1.dot_product(b0)/norm(b0)**2) #projection of b1 on b0
    bstar1 = (b1 - mu*b0).n()   #1st gs vector (bstar0 = b0)
    sy = t
    py = sy.dot_product(bstar1)/bstar1.dot_product(bstar1) #projection on first gsvect
    ry = ceil( min( r/norm(bstar1), count ) ) #bound on the largest x-ennum
    px = (t.dot_product(b0)/b0.dot_product(b0)).n()
    maxrx = RR( sqrt( r**2 - (norm(bstar1)*round(py))**2 ) / norm(b0) ) #bound on the largest y-enum
    branch_lim = max( floor(ry), floor(maxrx) )
    retno = 0   #counter to determine when to stop
    for b in range( branch_lim ):
        for y in set( [ round(py+b) , round(py-b) ] ):#each y in that boundary has 2*(b-1)+1 enumerated children,
                                     #so we have to enum only 2 children at that stage
            sx = sy - y*b1
            rx = (r**2-round(py-y)**2*norm(bstar1).n()**2).n()
            if rx <0: #we don't need complex numbers in range()
                break
            rx = RR( sqrt( rx ) / norm(b0) ) #bound on radius for x (we could store this, but the bound is too high and we expect first ~100 vectors to succseed)
            rx = round(rx)
            px = (sx.dot_product(b0)/b0.dot_product(b0)).n()    #center of x's
            if b<2*rx: #x in: [(px-rx), round(px+rx)]
                yield vector((round(px+b),y))*B
                yield vector((round(px-b),y))*B
                retno+=2 #increase the number of returns
            if retno >= count:
                break
        if retno >= count:
            break

        for yy in range( 2*b+1 ):   #newly added y's have 2*b+1 children to be enumerated
            y = round( (-1)**((yy)%2)*(yy//2) + py )    #we start enumeration from center to border
            sx = sy - y*b1
            rx = (r**2-round(py-y)**2*norm(bstar1).n()**2).n()  #bound on radius for x
            if rx <0:
                continue
            rx = RR( sqrt( rx ) )
            px = (sx.dot_product(b0)/b0.dot_product(b0)).n() #center of x's
            for xx in range( min( ceil(RR(2*rx)), 2*b+1 ) ): #x in: [(px-rx), round(px+rx)]
                x =  round( (-1)**((1+xx)%2)*xx + px ) #? (1+xx)%2 ?
                yield vector((x,y))*B
                retno+=1

def generate_close_vectors_my(lattice_basis, target, p, L, count=2000):
    """
    Generate a generator of vectors which are close, without
    bound determined by N to the `target`. The first
    element of the list is the solution of the CVP.

    This is included in this file just to show, how generate_close_vectors would look like after my fix. Tested.
    """
    # Compute the closest element
    lattice_basis =  matrix(lattice_basis).LLL()
    closest = solve_closest_vector_problem(lattice_basis, target)
    yield closest

    # Set the distance
    diff = target - closest
    distance = diff.dot_product(diff)

    # Compute the bound from L
    tar=[float(vv) for vv in target]
    b0 = L // p
    bound = floor((b0 + distance) + (2 * (b0 * distance).sqrt()))

    C = enum(lattice_basis,target, bound)
    closest_vectors = C
    for cc in closest_vectors:
        yield cc

def generate_close_vectors_canon(lattice_basis, target, bound, count=2000):
    """
    Generate a generator of vectors which are close, without
    bound determined by N to the `target`. The first
    element of the list is the solution of the CVP.
    """
    # Compute the closest element
    closest = solve_closest_vector_problem(lattice_basis, target)
    yield closest

    # Now use short vectors below a bound to find
    # close enough vectors

    short_vectors = generate_short_vectors(lattice_basis, bound, count=count)

    for v in short_vectors:
        yield closest + v

args = sys.argv

graph_flag = False
try:
    graph_flag = args[1] #the zeroth arg is the progname
    while graph_flag[0]=='-':
        graph_flag=graph_flag[1:]
    if graph_flag == "plot":
        graph_flag = True
except IndexError:
    pass

#load the retrieved lattices

filename = f"cvp_"
rootdir = "./lat/"
regex = re.compile(filename+"*")
data_folder = Path(rootdir)

objects = []
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            # print(file, end=", ")
            file_to_open = data_folder / file
            with open(file_to_open,"rb") as file:
                D = pickle.load( file )
                objects.append( D )

#For TMP in loaded lattices, compare.

counter = 0
for TMP in objects:
    bound, basis, target = TMP["bound"], TMP["basis"], TMP["target"]

    gen_my_implementation = enum(basis,target,bound)
    gen_canon_implementation = generate_close_vectors_canon(basis, target, bound)

    lmy = list(islice(gen_my_implementation,64))    #list of enum vect (my)
    lcanon = list(islice(gen_my_implementation,64))   #list of enum vect (their)
    nmy = [ norm(v).n(40) for v in lmy ]
    ncanon = [ norm(v).n(40) for v in lcanon ]
    diff = [nmy[i]-ncanon[i] for i in range(min(len(nmy),len(ncanon)))]
    try:
        multdiff = [abs(nmy[i]/ncanon[i]) for i in range(min(len(nmy),len(ncanon)))]
    except ZeroDivisionError:
        multdiff = "Zero Division Error: one of the close vectors is zero."
    print( f"My: {nmy}" )
    print(f"They: {ncanon}")
    print(f"Diff: {diff}")
    print(f"Mult diff: {multdiff}")
    if graph_flag:
        # list_plot( [RR(d) for d in diff] ).save_image(f"{counter}_diff.png")
        plt.plot(np.array(nmy), color='blue')
        plt.plot(ncanon, color='red')
        red_patch = mpatches.Patch(color='blue', label='My')
        blue_patch = mpatches.Patch(color='red', label='SQISign')
        plt.legend(handles=[blue_patch,red_patch])
        plt.savefig(f"{counter}_comp.png")
        plt.clf()
        try:
            if isinstance( multdiff, list ):
                list_plot( multdiff ).save_image(f"{counter}_mult.png")
        except ValueError:
            print("What?")
    counter+=1
