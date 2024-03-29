#!/usr/local/bin/python3
import numpy as np


def PolicyIteration( problem ):
    # 1. Initialization
    #       arbitrarily V and PI for all  s ∈ S
    V, PI, Gamma = problem.Initialization()  # V[s] should give state value
    theta = 0.0001

    nPolicyCount = 0
    # policy iteration loop
    while True: 
        # 2. Policy Evaluation
        # Loop
        nSweepCount = 0
        while True:  # evaluation , til converge
            delta = 0  # Δ ← 0
            V_prime = V.copy()  # backup, synchronized iteration
            # Loop for each s ∈ S
            for s in problem.AllStates():
                if problem.IsTerminal(s): continue
                # v ← V(s)
                v = V_prime[s]   

                # V(s) <- ∑ p( s',r | s, π(s )[ r+ gamma·V(s') ] 
                # calue Vπ, but to deal with tie break, 
                actions = problem.GetPolicy( s )  # here follows π , there may be multiple
                                                  # equally optimal actions under π
                # normally you need calc Qπ for control purpose later
                q_values = np.zeros( len(actions) )
                for i,action in enumerate(actions):
                    q_val = 0
                    for p, r, s_prime in problem.Successor( s, action ):
                        q_val += p * ( r + Gamma * V_prime[ s_prime ] )
                    q_values[i] = q_val
                # get new value
                # only keep .7 for float number , actions are evenly distributed
                V[s] = round( q_values.mean(), 7 )

                # Δ ← max( Δ , |v-V(s)| )
                delta = max( delta, abs( v - V[s] ) )

            nSweepCount += 1
            # until  Δ<θ (a small positive number determining the accuracy of estimation)
            print( "debug: evaluation {}th sweep, delta: {}".format( nSweepCount, delta ))
            if delta < theta:
                break
            # if True: return #DEBUG

        print( "debug: evaluation converge afte {} sweep".format( nSweepCount ))

        # 3. Policy Improvement
        nPolicyCount += 1

        bPolicyStable = True # policy-stable  ← true
        for s in problem.AllStates():  # for each s ∈ S
            if problem.IsTerminal(s): continue

            old_actions = PI[s]  # old-action ← π(s)
              
            actions = problem.AvailableActions( s )  # here check all available actions
            q_values = np.zeros( len(actions) )
            for i,action in enumerate(actions):
                q_val = 0
                for p, r, s_prime in problem.Successor( s, action ):
                    q_val += p * ( r + Gamma * V[ s_prime ] )  # here should be V, not V_prime
                q_values[i] = q_val
            # π(s) = argmax ( argmax may cause never terminate if the policy switch between 
            #   2 or more polices that are equally good. return a set of max actions instead )
            PI[s] = tuple([ v for i,v in enumerate(actions) if i in np.where( q_values == q_values.max() )[0] ]) 
            # if old-action != π(s), then policy-stable ← false
            if old_actions != PI[s]:
                bPolicyStable = False
        
        print( "π{}".format( nPolicyCount ) )

        # If policy-stable, then stop and return
        if bPolicyStable:
            return V,PI
        # if True: return  #DEBUG test Small GridWorld
            


if __name__ == "__main__":
    from problemCarRental import Problem
    # from problemGridworld import Problem

    problem = Problem()
    PolicyIteration( problem )
    problem.Show_cli()



