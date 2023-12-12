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

import numpy as np

def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

RR = RealField(100)

def strong_approximation_lattice_heuristic_emulator(N, C, D, λ, L, small_power_of_two=False, seed=0, which_enum="my"):
    """
    Taken from KLPT.py - strong_approximation_lattice_heuristic
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
        close_vectors = generate_close_vectors_my(lattice_basis, -target, p, L, seed=seed, which_enum=which_enum, count=2000, dump=False)
    else:
        close_vectors = generate_close_vectors_old(lattice_basis, -target, p, L, seed=seed, which_enum=which_enum, count=2000, dump=False)

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
    print("\n")
    if xp is None:
        # Never found vector which had a valid solution
        return None, None

    # Use solution to construct element μ
    # μ = λ*j*(C + D*ω) + N*(xp + ω*yp + j*(zp + ω*tp))

    # If we swapped earlier, swap again!
    if swap:
        C, D = D, C
        tp, zp = zp, tp

    μ = N * xp + N * yp * ω + (λ * C + N * zp) * j + (λ * D + N * tp) * j * ω

    # Check that Nrd(μ) == L
    # and that μ is in O0
    rednrm = μ.reduced_norm()
    assert rednrm == L
    assert μ in O0
    return stepnum, μ #μ

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
            file_to_open = data_folder / file
            with open(file_to_open,"rb") as file:
                D = pickle.load( file )
                objects2.append( D )

objects.sort(key=lambda obj:obj["seed"])
objects2.sort(key=lambda obj:obj["seed"])

#For TMP in loaded lattices, compare.
normrat_list, steprat_list = [], []

counter, mysucc, oldsucc = 0, 0, 0
for ii in range(len(objects)):
    TMP = objects[ii]
    TMP2 = None
    for ITER in objects2:
        if ITER["seed"] == TMP["seed"]:
            TMP2 = ITER
            break
    if TMP2 is None:
        continue
    bound, basis, target, p, L = TMP["bound"], TMP["basis"], TMP["target"], TMP["p"], TMP["L"] #retrieve all the info about lattices
    print( f"bound, basis, target, p, L: {bound, basis, target, p, L}" )

    gen_my_implementation = enum(basis,-target, bound)  #sample_babai( basis, -target, (bound)**0.5/2, count=144 ) 
    gen_canon_implementation = generate_close_vectors_old(basis, -target, p, L, count=144, dump=False) #note, the target here is with plus sign!

    N, C, D, λ, L, small_power_of_two = TMP2["N"], TMP2["C"], TMP2["D"], TMP2["lambda"], TMP2["L"], TMP2["small_power_of_two"]
    try:
        mystepnum, myrednrm = strong_approximation_lattice_heuristic_emulator(N, C, D, λ, L, small_power_of_two, which_enum="my")
        myrednrm = -1 if myrednrm is None else vector( list(myrednrm) ).norm().n()
        if myrednrm > 0:
            mysucc+=1
    except AssertionError:
        mystepnum, myrednrm = -1, -1
    try:
        oldstepnum, oldrednrm = strong_approximation_lattice_heuristic_emulator(N, C, D, λ, L, small_power_of_two, which_enum="not_my")
        oldrednrm = -1 if oldrednrm is None else vector( list(oldrednrm) ).norm().n()
        if oldrednrm > 0:
            oldsucc+=1
    except AssertionError:
        oldstepnum, oldrednrm = -1, -1
    if mystepnum is None:
        mystepnum, myrednrm = -1, -1
    if oldstepnum is None:
        oldstepnum, oldrednrm = -1, -1
    msg = f"failed to find"
    print( f"My result:      { msg if mystepnum<0 else mystepnum } steps with norm { msg if myrednrm<0 else myrednrm }" )
    t0, minLen, _, _, _, _ = NP( basis, -target )
    print( f"Classic result: { msg if oldstepnum<0 else oldstepnum }  steps  with norm { msg if oldrednrm<0 else oldrednrm } ||b0-t||={norm(target+t0).n(), minLen**0.5}" )
    normrat = myrednrm/oldrednrm
    steprat = mystepnum/oldstepnum
    if steprat>0 and mystepnum>0:
        print( f"Ratio norm diff: my/classic: {(normrat)} mysteps/classicsteps: {RealField(40)(steprat)}" )
        normrat_list.append(normrat)
        steprat_list.append(steprat)
    print()


    lmy = list(islice(gen_my_implementation,nsamples))    #list of enum vect (my)
    lcanon = list(islice(gen_canon_implementation,nsamples))   #list of enum vect (their)
    nmy = [ norm(v-target).n(40) for v in lmy ]
    ncanon = [ norm(v-target).n(40) for v in lcanon ]
    diff = [nmy[i]-ncanon[i] for i in range(min(len(nmy),len(ncanon)))]
    try:
        multdiff = [abs(nmy[i]/ncanon[i]) for i in range(min(len(nmy),len(ncanon)))]
    except ZeroDivisionError:
        multdiff = "Zero Division Error: one of the close vectors is zero."
    # print( f"My: {nmy}" )
    # print(f"They: {ncanon}")
    # print(f"Diff: {diff}")
    # print(f"Mult diff: {multdiff}")
    Lat = IntegerMatrix.from_matrix( basis )
    G = GSO.Mat( Lat )
    G.update_gso()

    # print(f"debug: {nmy} \n{ncanon}")
    if graph_flag:
        # list_plot( [RR(d) for d in diff] ).save_image(f"{counter}_diff.png")
        plt.plot(np.array(nmy), color='blue')
        plt.plot(ncanon, color='red')

        if mystepnum>0:
            plt.plot([mystepnum],[nmy[min(len(nmy)-1,mystepnum)]], marker="*",color="blue", markersize=15,  markeredgewidth=2, markeredgecolor=(0,0,0, 1)) #color="black"

        if oldstepnum>0:
            plt.plot([oldstepnum],[ncanon[min(max(0, len(ncanon)-1),oldstepnum)]], marker="*",color="red", markersize=15, markeredgewidth=2, markeredgecolor=(0,0,0, 1))
        red_patch = mpatches.Patch(color='blue', label='My')
        # black_patch = mpatches.Patch(marker = "*", color='black', label='My - found')
        blue_patch = mpatches.Patch(color='red', label='SQISign')
        # yellow_patch = mpatches.Patch(marker = "*", color='red', label='SQISign - found') #
        plt.legend(handles=[blue_patch,red_patch])

        plt.title(f"r11/r00: {G.get_r(1,1)**0.5 / G.get_r(0,0)**0.5: .4f}, mu10: {G.get_mu(1,0): .4f}, bnd/r00^2: {bound/ G.get_r(0,0): .4f}")

        plt.savefig(f"{counter}_comp.png")
        plt.clf()
        print(f"my: {mystepnum} | their: {oldstepnum}")
        # try:
        #     if isinstance( multdiff, list ):
        #         list_plot( multdiff ).save_image(f"{counter}_mult.png")
        # except ValueError:
        #     print("What?")
    counter+=1

print( f"Mean step ratio: {geo_mean_overflow(steprat_list)}" )
print( f"Mean norm ratio: {geo_mean_overflow(normrat_list)}" )
print( f"My successes: {mysucc}. Old successes: {oldsucc} out of {counter}" )
