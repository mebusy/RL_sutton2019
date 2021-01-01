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

    def AvailableActions(self, state ):
        nCar1, nCar2 = state
        # in this problem, you can move upto 5 cars between two locations.
        upper = min( nCar1 , MAX_CAR_MOVED , MAX_CAR_AVAILABLE - nCar2  )
        lower = -min( nCar2 , MAX_CAR_MOVED , MAX_CAR_AVAILABLE - nCar1  )
        return list( range( lower, upper + 1 ) )

    def GetPolicy(self, state ):
        return self.PI[ state ]

    # return (  ( prob, reward, state'   ),  )
    @functools.lru_cache(maxsize=None)
    def Successor(self, state, action ):
        nCar1, nCar2 = state
        r0 = -2 * abs(action)
        # moving cars
        nCar1 = min( nCar1 - action, MAX_CAR_AVAILABLE )
        nCar2 = min( nCar2 + action, MAX_CAR_AVAILABLE )

        sucs = []
        p_sum = 0
        for nCar1_prime , nCar2_prime in itertools.product( range( self.V.shape[0]), range( self.V.shape[1]  ) ):
            p, r = self.reward( (nCar1, nCar2), ( nCar1_prime , nCar2_prime ) )
            p_sum += p
            sucs.append( [p, r0+r , ( nCar1_prime , nCar2_prime ) ] )

        if p_sum != 0:
            for suc in sucs:
                suc[0] /= p_sum

        # print(suc)
        return sucs

    # probability in loc1 , if nCar delta = ?
    @functools.lru_cache(maxsize=None)
    def probRewardLoc1Delta(self, delta, nCar ):
        nIter = min( nCar, 11 )
        p = [ stats.poisson.pmf( i,3 )* stats.poisson.pmf( i-delta,3 )  for i in range(nIter) ]         
        return sum(p) , sum( [ p[i]*i*10 for i in range(nIter) ] )

    # probability in loc2 , if nCar delta = ?
    @functools.lru_cache(maxsize=None)
    def probRewardLoc2Delta(self, delta, nCar ):
        nIter = min( nCar, 11 )
        p = [ stats.poisson.pmf( i,4 )* stats.poisson.pmf( i-delta,2 )  for i in range(nIter) ]         
        return sum(p) , sum( [ p[i]*i*10 for i in range(nIter) ] )


    def reward( self, state, state_prime ):
        nCar1, nCar2 = state
        nCar1_prime , nCar2_prime = state_prime
        
        delta1 = nCar1_prime - nCar1
        p1, r1 = self.probRewardLoc1Delta( delta1, nCar1 )

        delta2 = nCar2_prime - nCar2
        p2, r2 = self.probRewardLoc2Delta( delta2, nCar2 )
        return p1*p2, r1+r2


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



        

