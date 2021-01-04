#!/usr/local/bin/python3
import numpy as np
import itertools
from collections import defaultdict
from  scipy import stats
import functools

MAX_CAR_MOVED = 5
MAX_CAR_AVAILABLE = 20

class Problem(object):
    def __init__(self):
        pass

    def Initialization(self):
        self.V = np.zeros( [MAX_CAR_AVAILABLE+1,MAX_CAR_AVAILABLE+1] )  # np array
        self.PI = defaultdict( lambda: (0,) )   # policy is a tuple
        self.Gamma = 0.9
        
        # V[s] should give state value
        # V[s] gives policy
        return self.V, self.PI, self.Gamma
        
    def AllStates(self):
        return itertools.product( range( self.V.shape[0]), range( self.V.shape[1]  ) )

    def IsTerminal(self,state):
        return False

    def AvailableActions(self, state ):
        nCar1, nCar2 = state
        # in this problem, you can move upto 5 cars between two locations.
        upper = min( nCar1 , MAX_CAR_MOVED  )
        lower = -min( nCar2 , MAX_CAR_MOVED  )
        return list( range( lower, upper + 1 ) )

    def GetPolicy(self, state ):
        return self.PI[ state ]

    @functools.lru_cache(maxsize=None)
    def pmf(self, k, loc ):
        return stats.poisson.pmf( k,loc )

    @functools.lru_cache(maxsize=None)
    def Successor(self, state, action ):
        nCar1, nCar2 = state
        r0 = -2 * abs(action)
        # moving cars
        nCar1 = min(nCar1 - action, MAX_CAR_AVAILABLE)
        nCar2 = min(nCar2 + action, MAX_CAR_AVAILABLE)

        assert nCar1 >=0 and nCar1 <= MAX_CAR_AVAILABLE
        assert nCar2 >=0 and nCar2 <= MAX_CAR_AVAILABLE

        state_pr = defaultdict( lambda: [0,0] )
        MAX_REQ = 11

        for req1 in range(MAX_REQ):
            rent1 = min( req1, nCar1 )
            for ret1 in range(MAX_REQ):
                nCar1_prime = min(nCar1 - rent1 + ret1, MAX_CAR_AVAILABLE)
                p1 = self.pmf( req1, 3 ) * self.pmf( ret1, 3 )
                if p1 < 0.00001:
                    continue

                for req2 in range(MAX_REQ):
                    rent2 = min( req2, nCar2 )
                    for ret2 in range(MAX_REQ):
                        nCar2_prime = min(nCar2 - rent2 + ret2, MAX_CAR_AVAILABLE)
                        p2 = self.pmf( req2, 4 ) * self.pmf( ret2, 2 )
                        if p2 < 0.00001:
                            continue
                            
                        s_prime = ( nCar1_prime, nCar2_prime ) 
                        state_pr[ s_prime ] [0] += p1*p2
                        state_pr[ s_prime ] [1] += ( rent1 + rent2) * 10 * p1*p2

        sucs = []
        p_sum = 0
        for s_prime , v in state_pr.items():
            p,r = v 
            if p < 0.000001 : continue
            p_sum += p
            sucs.append([ p, r0 + r/p, s_prime] )

        assert p_sum >= 0.95 and p_sum <= 1.0 , "{},suc total prob:{}".format( state, p_sum )
        return sucs


    def Show_cli(self):
        out = []
        for s1 in range( MAX_CAR_AVAILABLE , -1, -1):
            for s2 in range( MAX_CAR_AVAILABLE + 1):
                # print (s1, s2)
                val = self.PI[(s1, s2)][0]
                out.append(  "%2d" % val  )
            out.append('\n')

        out = ''.join(out)
        print(out)


        out = []
        for s1 in range( MAX_CAR_AVAILABLE , -1, -1):
            for s2 in range( MAX_CAR_AVAILABLE + 1):
                # print (s1, s2)
                val = self.V[(s1, s2)]
                out.append(  "%4d" % int(val)  )
            out.append('\n')

        out = ''.join(out)
        print(out)


if __name__ == "__main__":
    nSample = 1000000
    requests1 = np.random.poisson(3, nSample)
    returns1 = np.random.poisson( 3, nSample)
    requests2 = np.random.poisson(4, nSample)
    returns2 = np.random.poisson( 2, nSample)

    for _nCar1,_nCar2,action,nCar1_prime, nCar2_prime in \
        [ [16,16,2, 14,18], [ 1,1,-1,2,0 ], [ 20,20,4 , 18,19 ] ]:
        cnt = 0
        r = 0
        print( "state-action:", _nCar1,_nCar2,action )
        nCar1 = min( _nCar1 - action, MAX_CAR_AVAILABLE )
        nCar2 = min( _nCar2 + action, MAX_CAR_AVAILABLE )
 
        for i in range(nSample):
            rent1 = min( requests1[i], nCar1 )
            rent2 = min( requests2[i], nCar2 )

            _car1 = min(nCar1-rent1+returns1[i], MAX_CAR_AVAILABLE)
            _car2 = min(nCar2-rent2+returns2[i], MAX_CAR_AVAILABLE)
            if _car1 == nCar1_prime and _car2 == nCar2_prime:
                cnt += 1
                r += (rent1+rent2) * 10
        print( "sampled  :", cnt / nSample, -2*abs(action) + r / cnt )

        problem = Problem()
        problem.Initialization()
        sucs = problem.Successor( (_nCar1, _nCar2), action )
        for p,r,s_prime in sucs:
            if s_prime == ( nCar1_prime  , nCar2_prime ):
                print( "algorihtm:", p,r )

        

