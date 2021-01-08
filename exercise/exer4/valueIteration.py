#!/usr/local/bin/python3
import numpy as np


def ValueIteration( problem ):
    # 1. Initialization
    #       arbitrarily V and PI for all  s ∈ S, except that V (terminal)=0
    V, PI, Gamma = problem.Initialization()  # V[s] should give state value
    theta = 0.0001

    # Loop
    nSweepCount = 0
    while True:  # evaluation , til converge
        delta = 0  # Δ ← 0   Δ control the entire loop, while in policy iteration Δ  only control evaluation loop
        V_prime = V.copy()  # backup, synchronized iteration
        # Loop for each s ∈ S
        for s in problem.AllStates():
            if problem.IsTerminal(s): continue
            # v ← V(s)
            v = V_prime[s]   

            # V(s) <- max_a ∑ p( s',r | s, π(s )[ r+ gamma·V(s') ] 
            actions = problem.AvailableActions( s )  # here check all available actions
            q_values = np.zeros( len(actions) )
            for i,action in enumerate(actions):
                q_val = 0
                for p, r, s_prime in problem.Successor( s, action ):
                    q_val += p * ( r + Gamma * V_prime[ s_prime ] )  # V_prime here
                q_values[i] = q_val

            # get new value
            # only keep .7 for float number
            V[s] = round( q_values.max(), 7 )

            # Δ ← max( Δ , |v-V(s)| )
            delta = max( delta, abs( v - V[s] ) )

        nSweepCount += 1
        # until  Δ<θ (a small positive number determining the accuracy of estimation)
        print( "debug: evaluation {}th sweep, delta: {}".format( nSweepCount, delta ))
        if delta < theta:
            break

    print( "debug: value converge afte {} sweep".format( nSweepCount ))

    # Output a deterministic policy, π≈π* , such that
    for s in problem.AllStates():
        if problem.IsTerminal(s): continue

        # π(s) = argmax_a ∑ p( s',r | s, π(s )[ r+ gamma·V(s') ] 
        actions = problem.AvailableActions( s )  # here check all available actions
        q_values = np.zeros( len(actions) )
        for i,action in enumerate(actions):
            q_val = 0
            for p, r, s_prime in problem.Successor( s, action ):
                q_val += p * ( r + Gamma * V_prime[ s_prime ] )  # V_prime here
            q_values[i] = q_val
        # PI[s] = (q_values.argmax(),)
        PI[s] = tuple([ v for i,v in enumerate(actions) if i in np.where( q_values == q_values.max() )[0] ]) 

    return V,PI


if __name__ == "__main__":
    from problemCarRental import Problem
    # from problemGridworld import Problem

    problem = Problem()
    ValueIteration( problem )
    problem.Show_cli()



