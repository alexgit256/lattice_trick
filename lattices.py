"ld""""
Functions which solve lattice problems for use in subalgorithms of KLPT.

For KLPT there are a few spots where we need to enumerate short vectors
of lattices to ensure the smallest possible solutions to Diophantine equations.

Namely, we need close vectors to a special lattice for the strong approximation
to ensure the output bound is ~pN^3. This is accomplished with GenerateCloseVectors.
We use FPYLLL for the underlying lattice computations, which seem to outperform
Pari. We also have the ability to enumerate rather than precompute all vectors,
which is better than Pari's qfminim.

For the generation of equivalent prime norm ideals, we have an ideal basis and
we find short norm vectors of this and immediately output algebra elements.
There's probably ways to reuse the GenerateShortVectors, but there's a few
things about the prime norm elements which require special treatment so we
chose to suffer code duplication for clearer specific functions.
"""

# Sage Imports
from sage.all import vector, floor, ZZ, Matrix, randint

# fpylll imports
import fpylll
from fpylll import IntegerMatrix, CVP
from fpylll.fplll.gso import MatGSO
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ, FPLLL, Enumeration, CVP, SVP
from fpylll.fplll.enumeration import EvaluatorStrategy

import time, pickle
from math import sqrt
import numpy as np

from sage import all
from sage.all_cmdline import *   # import sage library

def solve_closest_vector_problem(lattice_basis, target):
    """
    Use the fpylll library to solve the CVP problem for a given
    lattice basis and target vector
    """
    L = IntegerMatrix.from_matrix(lattice_basis.LLL())
    v = CVP.closest_vector(L, target)
    # fpylll returns a type `tuple` object
    return vector(v)


def generate_short_vectors_fpyll(L, bound, count=3000):
    """
    Helper function for GenerateShortVectors and
    generate_small_norm_quat which builds an iterator
    for short norm vectors of an LLL reduced lattice
    basis.
    """
    # # Move from Sage world to Fypll world
    A = IntegerMatrix.from_matrix(L)

    # Gram-Schmidt Othogonalization
    G = MatGSO(A)
    _ = G.update_gso()

    # Enumeration class on G with `count`` solutions
    # BEST_N_SOLUTIONS:
    # Starting with the nr_solutions-th solution, every time a new solution is found
    # the enumeration bound is updated to the length of the longest solution. If more
    # than nr_solutions were found, the longest is dropped.
    E = fpylll.Enumeration(
        G, nr_solutions=count, strategy=fpylll.EvaluatorStrategy.BEST_N_SOLUTIONS
    )

    # We need the row count when we call enumerate
    r = L.nrows()

    # If enumerate finds no solutions it raises an error, so we
    # wrap it in a try block
    try:
        # The arguments of enumerate are:
        # E.enumerate(first_row, last_row, max_dist, max_dist_expo)
        short_vectors = E.enumerate(0, r, bound**2, 0)
    except Exception as e:
        print(f"DEBUG [generate_short_vectors_fpyll]: No short vectors could be found...")
        print(f"{e}")
        short_vectors = []

    return short_vectors

def generate_short_vectors(lattice_basis, bound, count=3000):
    """
    Generate a generator of short vectors with norm <= `bound`
    returns at most `count` vectors.

    Most of the heavy lifting of this function is done by
    generate_short_vectors_fpyll
    """
    L = lattice_basis.LLL()
    short_vectors = generate_short_vectors_fpyll(L, bound, count=count)
    for _, xis in short_vectors:
        # Returns values x1,x2,...xr such that
        # x0*row[0] + ... + xr*row[r] = short vector
        v3 = vector([ZZ(xi) for xi in xis])
        v = v3 * L
        yield v

def norm_(l):
    return ( sum( ll**2 for ll in l )**0.5 )

def dot(v,w):
    return [ v[0]*w[0], v[1]*w[1] ]

# def EnumerateCloseVectors(G,q,target,bound):
#     bound = np.float128( bound )
#     B = G.B
#     a = np.float128( B[0][0]**2+q*B[0][1]**2 )
#     b = np.float128( 2*(B[0][0]*B[1][0] + B[0][1]*B[1][1]) )
#     c = np.float128( B[1][0]**2+q*B[1][1]**2 )
#     assert( 4*a*c-b**2 > 0 ), f"Discriminant!"
#
#     bndy = int( floor( (sqrt(4*a**2*bound)+1)/sqrt(4*a*c-b**2) ) + 1 )
#     r00, r01, r11 = G.get_r(0,0)**0.5, G.get_mu(1,0), G.get_r(1,1)**0.5
#     # R = ([
#     #     [ r00, r01 ],
#     #     [ r01, r11 ]
#     # ])
#     R = np.ndarray([2,2])
#     v = np.ndarray( [2] )
#     v[0], v[1] = target[0], target[1]
#     R[0,0], R[0,1] = r00, r01
#     R[1,0], R[1,1] = r01, r11
#
#     xmid, ymid = v.dot( R**-1 )
#     print(R)
#     print(xmid, ymid)
#     xmid, ymid = round(xmid), round(ymid)
#     print(f"bndy: {bndy}")
#     for i in range(1,bndy):
#         y = ymid + (-1)**(i%2)*(i//2)
#         #bndx = ( 2*a*(1+sqrt(4*a**2*bound+4*c*a*np.float128(y)**2-b**2*y**2)-b*y*sqrt(4*a**3)) )
#         bndx = 100 #int( floor( -bndx/(2*a*sqrt(4*a**3)) )+1 )
#         print(f"bndx: {bndx}")
#         for j in range(1,bndx):
#             x = xmid + (-1)**(j%2)*(j//2)
#             yield (x,y)


def enum(B,t,r, count = 300):
    out = []
    b0, b1 = vector(B[0]), vector(B[1])
    mu = RR(b1.dot_product(b0)/norm(b0)**2)
    bstar1 = (b1 - mu*b0).n()
    sy = t
    py = sy.dot_product(bstar1)/bstar1.dot_product(bstar1) #projection on second gsvect
    ry = ceil( min( r/norm(bstar1), count ) ) #bound on the largest x-ennum
    px = (t.dot_product(b0)/b0.dot_product(b0)).n()
    maxrx = RR( sqrt( r**2 - (norm(bstar1)*round(py))**2 ) / norm(b0) ) #bound on the largest y-enum
    branch_lim = max( round(ry), round(maxrx) )
    retno = 0
    for b in range( branch_lim ):
        for y in set( [ round(py+b) , round(py-b) ] ):#each y in that boundary has 2*(b-1)+1 enumerated children,
                                     #so we have to enum only 2 children at that stage
            sx = sy - y*b1
            rx = (r**2-round(py-y)**2*norm(bstar1).n()**2).n()
            if rx <0:
                break
            rx = RR( sqrt( rx ) / norm(b0) )
            rx = round(rx)
            px = (sx.dot_product(b0)/b0.dot_product(b0)).n()
            if b<2*rx: #x in: round(px-rx), round(px+rx)
                yield vector((round(px+b),y))*B
                yield vector((round(px-b),y))*B
                retno+=2
            if retno >= count:
                break
        if retno >= count:
            break

        #for y in range(round(py)-b+1,round(py)+b): #newly added y's have 2*b+1
        for yy in range( 2*b+1 ):
            y = round( (-1)**((yy)%2)*(yy//2) + py )
            sx = sy - y*b1
            rx = (r**2-round(py-y)**2*norm(bstar1).n()**2).n()
            if rx <0:
                continue
            rx = RR( sqrt( rx ) )
            px = (sx.dot_product(b0)/b0.dot_product(b0)).n()
            for xx in range( min( ceil(RR(2*rx)), 2*b+1 ) ): #x in: round(px-rx), round(px+rx)
                x =  round( (-1)**((1+xx)%2)*xx + px )
                # print(b,x,y)
#                 print("ooo", (x-round(px),y-round(py)))
                yield vector((x,y))*B
                retno+=1

def generate_close_vectors_my(lattice_basis, target, p, L, count=200, seed=0, which_enum="my", dump=True):
    """
    Generate a generator of vectors which are close, without
    bound determined by N to the `target`. The first
    element of the list is the solution of the CVP.
    """
    # Compute the closest element
    lattice_basis =  matrix(lattice_basis).LLL()
    closest = solve_closest_vector_problem(lattice_basis, target)
    print(f"cl: {lattice_basis.solve_left(closest)}")
    yield closest

    # print(f"target: {target}")
    # Now use short vectors below a bound to find
    # close enough vectors

    # Set the distance
    diff = target - closest
    distance = diff.dot_product(diff)
    # print(target)

    # Compute the bound from L
    tar=[float(vv) for vv in target]
    b0 = L // p
    bound = floor((b0 + distance) + (2 * (b0 * distance).sqrt()))
    if dump:
        with open( f"cvp_{seed}", "wb" ) as file:
            pickle.dump({
                "basis": lattice_basis,
                "target": target,
                "bound": bound,
                "p": p,
                "L": L,
                "seed": seed
            }, file)

    C = enum(lattice_basis,target, bound, count=count)
    closest_vectors = C #[ tmp for tmp in sorted([c for c in C], key=(lambda v:norm(v-target).n())) ]
    for cc in closest_vectors:
        #print(f"nrm: {norm(cc).n()}", end=", ")
        yield cc

def generate_close_vectors_old(lattice_basis, target, p, L, count=1000, seed=0, which_enum="my", dump=True): #old ver
    """
    Generate a generator of vectors which are close, without
    bound determined by N to the `target`. The first
    element of the list is the solution of the CVP.
    """
    # Compute the closest element
    closest = solve_closest_vector_problem(lattice_basis, target)
    yield closest
    # print(f"target: {target}")
    # Now use short vectors below a bound to find
    # close enough vectors

    # Set the distance
    diff = target - closest
    distance = diff.dot_product(diff)
    # print(target)

    # Compute the bound from L
    tar=[float(vv) for vv in target]
    b0 = L // p
    bound = floor((b0 + distance) + (2 * (b0 * distance).sqrt()))

    B = IntegerMatrix.from_matrix( lattice_basis )
    G = GSO.Mat( B, float_type="ld" )
    G.update_gso()
    # seed = randint(0,2**20)
    if dump:
        with open( f"cvp_{seed}", "wb" ) as file:
            pickle.dump({
                "basis": lattice_basis,
                "target": target,
                "bound": bound,
                "p": p,
                "L": L,
                "seed": seed
            }, file)

    # lattice_basis = lattice_basis.LLL()
    # B = IntegerMatrix.from_matrix( lattice_basis )
    # G = GSO.Mat( B, float_type="ld" )
    # G.update_gso()

    # enum = Enumeration( G, strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS, nr_solutions=count )
    # radius, re = bound , 0
    # # print(f"bound: {bound*1.0}, {[norm_(ll) for ll in lattice_basis]}, renum= {b0} | seed = {seed} ")
    #
    # C = enum.enumerate( 0, G.B.nrows, 0.8*bound, re, target=diff  ) #target=[float(vv) for vv in target]
    #
    # t0 = time.perf_counter()
    # print(f"enum done in: {time.perf_counter()-t0}")
    # C = [ vector(ZZ,cc[1])*lattice_basis for cc in C ]
    # closest_vectors = [ tmp for tmp in sorted([c for c in C], key=(lambda v:norm_([v[k]-target[k] for k in range(len(v))])) ) ]
    # for cc in closest_vectors:
    #     yield cc+closest

    short_vectors = generate_short_vectors(lattice_basis, bound, count=count)

    for v in short_vectors:
        yield closest + v


def generate_small_norm_quat(Ibasis, bound, count=3000):
    """
    Given an ideal I and an upper bound for the scaled
    norm Nrd(a) / n(I), finds elements a ∈ B such that
    a has small norm.
    """
    # Before starting anything, just send out the basis
    # sometimes this works, and much quicker.
    for bi in Ibasis:
        yield bi

    # Recover Quaternion algebra from IBasis for use later
    B = Ibasis[0].parent()

    # Write Ibasis as a matrix
    Ibasis_matrix = Matrix([x.coefficient_tuple() for x in Ibasis]).transpose()

    # Can't do LLL in QQ, so we move to ZZ by clearing
    # the denominator
    lattice_basis, _ = Ibasis_matrix._clear_denom()
    L = lattice_basis.LLL()

    # Move from Sage world to Fypll world
    short_vectors = generate_short_vectors_fpyll(L, bound, count=count)

    for _, xis in short_vectors:
        # Returns values x1,x2,...xr such that
        # x0*row[0] + ... + xr*row[r] = short vector
        v3 = vector([ZZ(xi) for xi in xis])

        # Often the very shortest are all composite
        # this forces some of the latter element?
        # Decide with a coin-flip?
        if randint(0, 1) == 1:
            v3[2] = 1

        v = Ibasis_matrix * v3

        yield B(v)

    print(
        f"WARNING [generate_small_norm_quat]: "
        "Exhausted all short vectors, if you're seeing this SQISign is likely about to fail."
    )
