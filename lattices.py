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
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
RR = RealField( 144 )

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
        short_vectors = E.enumerate(0, r, bound, 0)
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

def list_generate_short_vectors(lattice_basis, bound, count=3000):
    """
    Generate a generator of short vectors with norm <= `bound`
    returns at most `count` vectors.

    Most of the heavy lifting of this function is done by
    generate_short_vectors_fpyll
    """
    L = lattice_basis.LLL()
    l = []
    short_vectors = generate_short_vectors_fpyll(L, bound, count=count)
    for _, xis in short_vectors:
        # Returns values x1,x2,...xr such that
        # x0*row[0] + ... + xr*row[r] = short vector
        v3 = vector([ZZ(xi) for xi in xis])
        v = v3 * L
        l.append( v )
    return(l)

def norm_(l):
    return ( sum( ll**2 for ll in l )**0.5 )

def dot(v,w):
    return [ v[0]*w[0], v[1]*w[1] ]

def NPfpylll( B, t ):
    Bint = IntegerMatrix.from_matrix( B )
    G = GSO.Mat( Bint )
    G.update_gso()
    ccl = G.babai( t )
    cl = vector( ZZ, Bint.multiply_left(ccl) )
    diff = t-cl
    return cl, diff.dot_product(diff)

def NP( B, t, q=1 ):
    t1, e1 = t, 0 #line 1
    assert q==1, "Not implemented for q>1 yet"
    r00 = B[0].dot_product(B[0])  #line 2 start
    mu10 = B[1].dot_product(B[0]) / r00
    b1star = B[1]-mu10*B[0]
    r11 = b1star.dot_product(b1star)  #line 2 end

    c1 = t1.dot_product( b1star ) / r11
    t0 = t1 - round(c1)*B[1]
    e0 = e1 + (c1-round(c1))**2*r11

    c0 = t0.dot_product( B[0] ) / r00
    tout = t0 - round(c0)*B[0]
    eout = e0 + (c0-round(c0))**2*r00

    return( vector( ZZ, t-tout ), eout, r00, r11, mu10, b1star )

def enum( B, t, R, count=2000, bnd=[lambda arg : 100, lambda arg : 0.5*arg], q=1 ):
    # r00 = B[0].dot_product(B[0])  #line 2 start
    # mu10 = B[1].dot_product(B[0]) / r00
    # b1star = B[1]-mu10*B[0]
    # r11 = b1star.dot_product(b1star)  #line 2 end
    R = R**2

    # R = R**2
    assert q==1, "Not implemented for q>1 yet"
    t0, minLen, r00, r11, mu10, b1star = NP( B, t, q=q ) #lines 2 & 3

    if minLen>R:
        print( f"No solution: {minLen} > {R}" )
        return None

    k = 1
    r = [ r00, r11 ]
    c = [ 0, 0 ]
#     T = [ t0, t ]
    e = [ 0, 0 ]
    Bstar = [ B[0], vector(b1star) ]

    Inty = RR( (bnd[1](R) / r[1])**0.5 ) #we need to enum every y in the worst case
    cstary = t.dot_product(Bstar[1]) / r[1] #center for y's
    cy = round( cstary ) #the rounded center for y coord
    # amount of steps to the left (resp. right) for y coordinate
    ly, ry = ceil(cstary-Inty/2), floor(cstary+Inty/2)
    print( f" ly, ry, cy, Inty : { ly, ry, cy, Inty }" )

    xcap = 30
    cntr = 0 #for auto abort
    goonflag = True
    for ystep in range( ceil(Inty/2) ):
            """
            Left edge.
            """
            if cy - ystep>=ly:
                c[1] = cy - ystep #go to the left edge
                # print(f"y: {c[1]}")
                t0 = t - c[1]*B[1]  #project target on line
                e[0] = e[1] + (c[1]-cstary)**2 * r[1]  #current squared-error

                Intx = min( xcap, (bnd[0](R - e[0]) / r[1])**0.5 ) #get len of x's for current y=c[1]
                cstarx = t0.dot_product(Bstar[0]) / r[0] #center for x's (for current y)
                cx = round( cstarx ) #the rounded center for x coord
                # amount of steps to the left (resp. right) for x coordinate
                lx, rx = floor(cstarx-Intx/2), ceil(cstarx+Intx/2)
                # print( f" lx, rx, cx, Intx : { lx, rx, cx, Intx }" )
                for xstep in range( ceil(Intx/2) ):
                    if cx-xstep>=lx: #if we are within the left boundary for x
                        c[0] = cx - xstep #go to the left edge
                        # print( c,end=", " )
                        # print(vector(c)*B,end=", ")
                        out = vector(c)*B
                        asrt = norm(out-t)**2 <= R
                        if not asrt:
                            print( norm(out-t)**2 / R )
                        assert asrt, f"Enum bad norm"
                        yield out
                        cntr+=1
                        if cntr>=count:
                            goonflag = False
                            break

                    if xstep!=0 and cx+xstep<=rx: #if we are within the right boundary for x and not at zero
                        c[0] = cx + xstep
                        out = vector(c)*B
                        asrt = norm(out-t)**2 <= R
                        if not asrt:
                            print( norm(out-t)**2 / R )
                        assert asrt, f"Enum bad norm"
                        yield out
                        cntr+=1
                        if cntr>=count:
                            goonflag = False
                            break
            """
            Right edge.
            """
            if ystep!=0 and cy - ystep <= ry:
                c[1] = cy + ystep #go to the left edge
                # print(f"y: {c[1]}")
                t0 = t - c[1]*B[1]  #project target on line
                e[0] = e[1] + (c[1]-cstary)**2 * r[1]  #current squared-error

                min( xcap, (bnd[0](R - e[0]) / r[1])**0.5 ) #get len of x's for current y=c[1]
                cstarx = t0.dot_product(Bstar[0]) / r[0] #center for x's (for current y)
                cx = round( cstarx ) #the rounded center for x coord
                # amount of steps to the left (resp. right) for x coordinate
                lx, rx = ceil(cstarx-Intx/2), floor(cstarx+Intx/2)
                # print( f" lx, rx, cx, Intx : { lx, rx, cx, Intx }" )
                for xstep in range( ceil(Intx/2) ):
                    if cx-xstep>=lx: #if we are within the left boundary for x
                        c[0] = cx - xstep #go to the left edge
                        # print(c,end=", ")
                        # print(vector(c)*B,end=", ")
                        out = vector(c)*B
                        asrt = norm(out-t)**2 <= R
                        if not asrt:
                            print( norm(out-t)**2 / R )
                        assert asrt, f"Enum bad norm"
                        yield out
                        cntr+=1
                        if cntr>=count:
                            goonflag = False
                            break

                    if xstep!=0 and cx+xstep<=rx: #if we are within the right boundary for x and not at zero
                        c[0] = cx + xstep
                        # print(c,end=", ")
                        out = vector(c)*B
                        asrt = norm(out-t)**2 <= R
                        if not asrt:
                            print( norm(out-t)**2 / R )
                        assert asrt, f"Enum bad norm"
                        yield out
                        cntr+=1
                        if cntr>=count:
                            goonflag = False
                            break
            if goonflag is False:
                break

def sample_babai( B, t, bound, count=2000 ):
    D = DiscreteGaussianDistributionIntegerSampler( sigma=bound )

    for cntr in range(count):
        t_ = t + vector( [D(), D()] )
        T, eout, r00, r11, mu10, b1star = NP( B, t_ )
        # print( f"R: {B.solve_left(T)}", end=", " )
        yield T

# def enum(B,t,r, count = 40):
#     out = []
#     b0, b1 = vector(B[0]), vector(B[1])
#     mu = RR(b1.dot_product(b0)/norm(b0)**2)
#     bstar1 = (b1 - mu*b0).n()
#     sy = t
#     py = sy.dot_product(bstar1)/bstar1.dot_product(bstar1) #projection on second gsvect
#     ry = ceil( min( r/norm(bstar1), count ) ) #bound on the largest x-ennum
#     px = (t.dot_product(b0)/b0.dot_product(b0)).n()
#     maxrx = RR( sqrt( r**2 - (norm(bstar1)*round(py))**2 ) / norm(b0) ) #bound on the largest y-enum
#     branch_lim = max( round(ry), round(maxrx) )
#     retno = 0
#     for b in range( branch_lim ):
# #         print("new")
#         for y in set( [ round(py+b) , round(py-b) ] ):#each y in that boundary has 2*(b-1)+1 enumerated children,
#                                      #so we have to enum only 2 children at that stage
#             sx = sy - y*b1
#             rx = (r**2-round(py-y)**2*norm(bstar1).n()**2).n()
#             if rx <0:
#                 break
#             rx = RR( sqrt( rx ) / norm(b0) )
#             rx = round(rx)
#             px = (sx.dot_product(b0)/b0.dot_product(b0)).n()
#             if b<2*rx: #x in: round(px-rx), round(px+rx)
#                 yield vector((round(px+b),y))*B
#                 yield vector((round(px-b),y))*B
# #                 print(b,"0 & 1",y)
# #                 print((round(b),y-round(py)))
# #                 print((round(-b),y-round(py)))
#                 retno+=2
#             if retno >= count:
#                 break
#         if retno >= count:
#             break
#
#         #for y in range(round(py)-b+1,round(py)+b): #newly added y's have 2*b+1
#         for yy in range( 2*b+1 ):
#             y = round( (-1)**((yy)%2)*(yy//2) + py )
#             sx = sy - y*b1
#             rx = (r**2-round(py-y)**2*norm(bstar1).n()**2).n()
#             if rx <0:
#                 continue
#             rx = RR( sqrt( rx ) )
#             px = (sx.dot_product(b0)/b0.dot_product(b0)).n()
#             for xx in range( min( ceil(RR(2*rx)), 2*b+1 ) ): #x in: round(px-rx), round(px+rx)
#                 x =  round( (-1)**((1+xx)%2)*xx + px )
#                 # print(b,x,y)
# #                 print("ooo", (x-round(px),y-round(py)))
#                 yield vector((x,y))*B
#                 retno+=1

def generate_close_vectors_my(lattice_basis, target, p, L, count=2000, seed=0, which_enum="my", dump=True):
    """
    Generate a generator of vectors which are close, without
    bound determined by N to the `target`. The first
    element of the list is the solution of the CVP.
    """
    # Compute the closest element
    lattice_basis =  matrix(lattice_basis).LLL()
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
    # bound = floor((b0 + distance) + (2 * (b0 * distance).sqrt()))
    bound = L//(2*p)
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
    print()

    C = enum(lattice_basis, target, bound, count=count, bnd=[lambda arg:0.5*arg, lambda arg:0.25*arg])
    # C = sample_babai( lattice_basis, target, bound=(bound)**0.5, count=count )
    closest_vectors = C #[ tmp for tmp in sorted([c for c in C], key=(lambda v:norm(v-target).n())) ]
    for cc in closest_vectors:
        #print(f"nrm: {norm(cc).n()}", end=", ")
        yield cc

def generate_close_vectors_old(lattice_basis, target, p, L, count=2000, seed=0, which_enum="my", dump=True): #old ver
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

    # B = IntegerMatrix.from_matrix( lattice_basis )
    # G = GSO.Mat( B, float_type="ld" )
    # G.update_gso()

    # yield next(EnumerateCloseVectors(G,1,target,bound))

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
