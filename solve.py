#!/usr/bin/venv python3
from sympy.stats import Normal, density, cdf, P
import time
import sympy as sy
import numpy as np



# alpha_L, h, S, mu, sigma^2, L
data = [
(0.99,	1.25,	250,	200,	6400,	2),	
(0.99,	1.25,	250,	150,	3600,	2),	
(0.99,	1,	    250,	200,	6400,	2),	
(0.99,	1,	    250,	150,	3600,	2),	
(0.95,	1.25,	250,	200,	6400,	2),	
(0.95,	1.25,	250,	150,	3600,	2),	
(0.95,	1,	    250,	200,	6400,	2),	
(0.95,	1,	    250,	150,	3600,	2),	
(0.99,	0.6,	250,	200,	6400,	2),	
(0.99,	0.6,	250,	150,	3600,	2),	
(0.99,	0.1,	250,	200,	6400,	2),	
(0.99,	0.1,	250,	150,	3600,	2),	
(0.95,	0.6,	250,	200,	6400,	2),	
(0.95,	0.6,	250,	150,	3600,	2),	
(0.95,	0.1,	250,	200,	6400,	2),	
(0.95,	0.1,	250,	150,	3600,	2),	
(0.99,	1.25,	250,	50,	    400,	2),	
(0.99,	1.25,	250,	10,	    16,	    2),	
(0.99,	1,	    250,	50,	    400,	2),	
(0.99,	1,	    250,	10,	    16,	    2),	
(0.95,	1.25,	250,	50,	    400,	2),	
(0.95,	1.25,	250,	10,	    16,	    2),	
(0.95,	1,	    250,	50,	    400,	2),	
(0.95,	1,	    250,	10,	    16,	    2),	
(0.99,	0.6,	250,	50,	    400,	2),	
(0.99,	0.6,	250,	10,	    16,	    2),	
(0.99,	0.1,	250,	50,	    400,	2),	
(0.99,	0.1,	250,	10,	    16,	    2),	
(0.95,	0.6,	250,	50,	    400,	2),	
(0.95,	0.6,	250,	10,	    16,	    2),	
(0.95,	0.1,	250,	50,	    400,	2),	
(0.95,	0.1,	250,	10,	    16,	    2)
]


def create_restriction(verbose = False):
    # Declare symbols
    alpha_L, r, mu, L, sigma, x, S, h, Q = sy.symbols('alpha_L r mu L sigma x S h Q')

    #D = Normal('x', mu*L, sigma*L)
    #sy.pprint(sy.simplify( P(D < r)) ) 
    Z = Normal(alpha_L, 0, 1)
    Z = cdf(Z)(alpha_L)
    Z = Z**(-1)
    # restriction <= 0
    restriction = mu * L + Z*((sigma * L)**0.5) - r
    restriction_2 = ((2*mu*S / h)**0.5) - Q
    if verbose:
        sy.pprint(restriction.evalf())
        sy.pprint(restriction_2.evalf())
    return restriction, restriction_2

def create_f_obj(verbose = False):
    ## Declare constants

    # Declare symbols
    S,mu,Q,r,L,h, x, sigma = sy.symbols('S mu Q r L h x sigma')
    # Functions to representate the objective function
    normal = Normal('x', 0 , 1)
    _cdf = cdf(normal)(x)
    _pdf = density(normal)(x)

    dist = 1
    H = (1/2) * ((x**2 + 1)*(1 - _cdf) - x * _pdf)
    # Replace the X on h with the value of inside H()
    H_replacement_1 = ((r  - mu*L) / (((sigma)*L)**(0.5)) )
    H_replacement_2 = ((Q + r  - mu*L) / (((sigma)*L)**(0.5)) )
    H_1 = (1/2) * ((x**2 + 1)*(1 - cdf(normal)(H_replacement_1).evalf() ) - x * density(normal)(H_replacement_1).evalf() )
    H_2 = (1/2) * ((x**2 + 1)*(1 - cdf(normal)(H_replacement_2).evalf() ) - x * density(normal)(H_replacement_2).evalf() )
    B = ((sigma * L)/Q) * (H.subs(x, H_replacement_1) - H.subs(x,H_replacement_2) )
    if verbose:
        sy.pprint(H_1)
        sy.pprint(H_2)
    f_obj = S*(mu/Q) + h * ((Q/2) + r - (mu * L) + B)
    if verbose:
        print("H Function")
        sy.pprint(H)
        print("B Function")
        sy.pprint(B)
        print("Objective function")
        sy.pprint( sy.simplify(f_obj.evalf()) )
    
    return f_obj


def do_KKT(f_obj, r1, r2, vals, verbose = False):

    # Create local symbols
    alpha_L, S,mu,Q,r,L,h, x, sigma, lambda_1, lambda_2 = sy.symbols('alpha_L S mu Q r L h x sigma lambda_1 lambda_2')
   
    # unpack the incoming data
    __aplha, __h, __S, __mu, __sigma, __L = vals
    # Create replacement vector and replace
    replacement = [ (alpha_L, __aplha), (h, __h), (S,__S), (mu, __mu), (sigma,__sigma), (L,__L) ]
    f_obj = f_obj.subs(replacement).evalf()
    r1 = r1.subs(replacement).evalf()
    r2 = r2.subs(replacement).evalf()

    # Get the lagang.
    lagang = f_obj + lambda_1*r1 + lambda_2*r2

    if verbose:
        sy.pprint("Langrangeano: ")
        sy.pprint(lagang)

    # Stationary conditions
    dl_dq = sy.diff(lagang, Q)
    dl_dr = sy.diff(lagang, r)

    # Complementary slack condition
    lambda_1_slack = lambda_1 * r1
    lambda_2_slack = lambda_2 * r2

    # primal factibility
    # r1
    # r2
    if verbose: 
        sy.pprint("Stationary conditions")
        sy.pprint(dl_dq)
        sy.pprint(dl_dr)
        sy.pprint("Complementary slack condition")
        sy.pprint(lambda_1_slack )
        sy.pprint(lambda_2_slack)
        sy.pprint("Primal factibility")
        sy.pprint(r1)
        sy.pprint(r2)
    

    if verbose:
        sy.pprint("Trying to solve")

    solution = sy.solve( (dl_dq, dl_dr, lambda_1_slack, lambda_2_slack) )
    Q_opt = solution[1][Q]
    r_opt = solution[1][r]
    z = f_obj.subs([(Q, Q_opt),(r, r_opt)]).evalf()
    if verbose:
        sy.pprint(z)
    sol = (vals, z, Q_opt, r_opt)
    return solution, sol


def solve(verbose = False):
    alpha_L, S,mu,Q,r,L,h, x, sigma, lambda_1, lambda_2 = sy.symbols('alpha_L S mu Q r L h x sigma lambda_1 lambda_2')
    f_obj = create_f_obj( verbose = verbose )
    r1, r2 = create_restriction( verbose = verbose )

    solutions = []
    for k in data:
        sol_, sol = do_KKT(f_obj, r1, r2, k, verbose = verbose )
        sy.pprint("Solution for {}".format( k ) )
        sy.pprint("Z: %f\tQ:%f\tr:%f" % (sol[1], sol[2], sol[3]))
        solutions.append(sol)
        sy.pprint("-------")

    print("Compute ended...")
    for vals, z, qopt, ropt in solutions:
        print("Case %s:\n\t Z: %f\tQ: %f\tr: %f\t" % (vals, z, qopt, ropt))


if __name__ == "__main__":
    #solve(verbose=True)
    try:
        startTime = time.time()
        solve()
        print("The entire process took %f seconds" % (time.time() - startTime ))
    except KeyboardInterrupt:
        print("\nExit flawlessly")
        exit(0)