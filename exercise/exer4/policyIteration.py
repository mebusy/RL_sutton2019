#!/usr/local/bin/python3
from problem import Problem
import numpy as np


def PolicyIteration( problem ):
    # 1. Initialization
    #       arbitrarily V and PI for all  s ∈ S
    V, PI, Gamma = problem.Initialization()  # V[s] should give state value
    theta = 0.01

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
                # v ← V(s)
                v = V_prime[s]   

                # V(s) <- ∑ p( s',r | s, π(s )[ r+ gamma·V(s') ] 
                actions = problem.GetPolicy( s )  # here follows π
                for action in actions:
                    v_tmp = 0
                    for p, r, s_prime in problem.Successor( s, action ):
                        v_tmp += p * ( r + Gamma * V_prime[ s_prime ] )
                    # get new value
                    V[s] = max( V[s], v_tmp )

                # Δ ← max( Δ , |v-V(s)| )
                delta = max( delta, abs( v - V[s] ) )

            nSweepCount += 1
            # until  Δ<θ (a small positive number determining the accuracy of estimation)
            print( "debug: evaluation {}th sweep, delta: {}".format( nSweepCount, delta ))
            if delta < theta:
                break
        print( "debug: evaluation converge afte {} sweep".format( nSweepCount ))

        # 3. Policy Improvement
        nPolicyCount += 1

        bPolicyStable = True # policy-stable  ← true
        for s in problem.AllStates():  # for each s ∈ S
            old_actions = PI[s]  # old-action ← π(s)
              
            actions = problem.AvailableActions( s )  # here check all available actions
            values = np.zeros( len(actions) )
            for i,action in enumerate(actions):
                v_tmp = 0
                for p, r, s_prime in problem.Successor( s, action ):
                    v_tmp += p * ( r + Gamma * V_prime[ s_prime ] )
                    if s[0] == 20 and s[1] == 16:
                        # print( s, action, "->",  p,r,s_prime )
                        pass
                values[i] = v_tmp
            # π(s) = argmax ( argmax may cause never terminate if the policy switch between 2 or more polices that are equally good. return a set of max actions instead )
            PI[s] = tuple([ v for i,v in enumerate(actions) if i in np.where( values == values.max() )[0] ]) 
            if s[0] == 20 and s[1] == 16:
                print( s, PI[s], actions, values)
            # if old-action != π(s), then policy-stable ← false
            if old_actions != PI[s]:
                bPolicyStable = False
        
        print( "π{}".format( nPolicyCount ) )

        # If policy-stable, then stop and return
        if bPolicyStable:
            return V,PI


if __name__ == "__main__":
    problem = Problem()
    PolicyIteration( problem )
    problem.show_cli()
