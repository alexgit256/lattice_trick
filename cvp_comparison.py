"""
To be launched as:
sage cvp_comparison.py [1> out.txt] [--plot]
"""

nsamples = 144

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

from lattices import *
from KLPT import *

RR = RealField(100)

def enum(B,t,r, count = 3000):
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

def generate_close_vectors_my_(lattice_basis, target, p, L, count=3000, dump=False):
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

def strong_approximation_lattice_heuristic_emulator(N, C, D, λ, L, small_power_of_two=False, seed=0, which_enum="my"):
    """
    Constructs a lattice basis and then looks for
    close vectors to the target.

    Allows for optimising output from pN^4 to pN^3,
    which helps keep the norm small and hence the
    degree of the isogenies small
    """
    # with open( f"salh_{seed}", "wb" ) as file:
    #     pickle.dump({
    #         "N": N,
    #         "C": C,
    #         "D": D,
    #         "lambda": λ,
    #         "small_power_of_two": small_power_of_two,
    #         "seed": seed
    #     }, file)

    # We really only expect this for the case when N = 2^k
    swap = False
    if D == 0 or gcd(D, N) != 1:
        C, D = D, C
        swap = True

    # Construct the lattice
    lattice_basis, target, zp0, tp0 = strong_approximation_construct_lattice(
        N, C, D, λ, L, small_power_of_two=small_power_of_two
    )

    # Generate vectors close to target
    if which_enum=="my":
        close_vectors = generate_close_vectors_my(lattice_basis, -target, p, L, seed=seed, which_enum=which_enum, dump=False)
    else:
        close_vectors = generate_close_vectors_old(lattice_basis, -target, p, L, seed=seed, which_enum=which_enum, dump=False)

    xp, yp = None, None
    stepnum = 0
    for close_v in close_vectors:
        stepnum += 1
        zp, tp = close_v
        assert zp % N == 0, "Can't divide zp by N"
        assert tp % N == 0, "Can't divide tp by N"

        zp = ZZ(zp / N) + zp0
        tp = ZZ(tp / N) + tp0
        M = L - p * quadratic_norm(λ * C + zp * N, λ * D + tp * N)
        M, check = M.quo_rem(N**2)
        assert check == 0, "Cant divide by N^2"

        if M < 0:
            continue

        # Try and find a solution to
        # M = x^2 + y^2
        two_squares = Cornacchia(ZZ(M), -ZZ(ω**2))
        if two_squares:
            xp, yp = two_squares
            break

    if xp is None:
        # Never found vector which had a valid solution
        return None

    # Use solution to construct element μ
    # μ = λ*j*(C + D*ω) + N*(xp + ω*yp + j*(zp + ω*tp))

    # If we swapped earlier, swap again!
    if swap:
        C, D = D, C
        tp, zp = zp, tp

    μ = N * xp + N * yp * ω + (λ * C + N * zp) * j + (λ * D + N * tp) * j * ω

    # Check that Nrd(μ) == L
    # and that μ is in O0
    assert μ.reduced_norm() == L
    assert μ in O0
    return stepnum #μ

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
rootdir = "./data/"
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

filename = f"salh_"
rootdir = "./data/"
regex = re.compile(filename+"*")
data_folder = Path(rootdir)

objects2 = []
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            # print(file, end=", ")
            file_to_open = data_folder / file
            with open(file_to_open,"rb") as file:
                D = pickle.load( file )
                objects2.append( D )

# assert len(objects)==len(objects2), f"Inconsistent data: different lens!"
objects.sort(key=lambda obj:obj["seed"])
objects2.sort(key=lambda obj:obj["seed"])
# assert all( [objects[i]["seed"]==objects2[i]["seed"] for i in range(len(objects))] ), f"Inconsistent data: seeds do not match!"

#For TMP in loaded lattices, compare.

counter = 0
for ii in range(len(objects)):
    TMP = objects[ii]
    TMP2 = None
    for ITER in objects2:
        if ITER["seed"] == TMP["seed"]:
            TMP2 = ITER
            break
    if TMP2 is None:
        continue
    bound, basis, target = TMP["bound"], TMP["basis"], TMP["target"]

    gen_my_implementation = enum(basis,target,bound)
    gen_canon_implementation = generate_close_vectors_canon(basis, target, bound)

    N, C, D, λ, L, small_power_of_two = TMP2["N"], TMP2["C"], TMP2["D"], TMP2["lambda"], TMP2["L"], TMP2["small_power_of_two"]
    try:
        mystepnum = strong_approximation_lattice_heuristic_emulator(N, C, D, λ, L, small_power_of_two, which_enum="my")
    except AssertionError:
        mystepnum = -1
    try:
        oldstepnum = strong_approximation_lattice_heuristic_emulator(N, C, D, λ, L, small_power_of_two, which_enum="not_my")
    except AssertionError:
        oldstepnum = -1
    if mystepnum is None:
        mystepnum=-1
    if oldstepnum is None:
        oldstepnum=-1



    lmy = list(islice(gen_my_implementation,nsamples))    #list of enum vect (my)
    lcanon = list(islice(gen_my_implementation,nsamples))   #list of enum vect (their)
    nmy = [ norm(v).n(40) for v in lmy ]
    ncanon = [ norm(v).n(40) for v in lcanon ]
    diff = [nmy[i]-ncanon[i] for i in range(min(len(nmy),len(ncanon)))]
    try:
        multdiff = [abs(nmy[i]/ncanon[i]) for i in range(min(len(nmy),len(ncanon)))]
    except ZeroDivisionError:
        multdiff = "Zero Division Error: one of the close vectors is zero."
    # print( f"My: {nmy}" )
    # print(f"They: {ncanon}")
    # print(f"Diff: {diff}")
    # print(f"Mult diff: {multdiff}")
    if graph_flag:
        # list_plot( [RR(d) for d in diff] ).save_image(f"{counter}_diff.png")
        plt.plot(np.array(nmy), color='blue')
        if mystepnum>0:
            plt.plot([mystepnum],[nmy[min(len(nmy)-1,mystepnum)]], marker="*",color="blue", markersize=15)
        plt.plot(ncanon, color='red')
        if oldstepnum>0:
            plt.plot([oldstepnum],[ncanon[min(len(ncanon)-1,oldstepnum)]], marker="*",color="red", markersize=15)
        red_patch = mpatches.Patch(color='blue', label='My')
        blue_patch = mpatches.Patch(color='red', label='SQISign')
        plt.legend(handles=[blue_patch,red_patch])
        plt.savefig(f"{counter}_comp.png")
        plt.clf()
        print(f"my: {mystepnum} | their: {oldstepnum}")
        # try:
        #     if isinstance( multdiff, list ):
        #         list_plot( multdiff ).save_image(f"{counter}_mult.png")
        # except ValueError:
        #     print("What?")
    counter+=1
