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
        upper = min( nCar1 , MAX_CAR_MOVED , MAX_CAR_AVAILABLE - nCar2  )
        lower = -min( nCar2 , MAX_CAR_MOVED , MAX_CAR_AVAILABLE - nCar1  )
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
        nCar1 = nCar1 - action
        nCar2 = nCar2 + action

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


    # return (  ( prob, reward, state'   ),  )
    @functools.lru_cache(maxsize=None)
    def Successor2(self, state, action ):
        nCar1, nCar2 = state
        r0 = -2 * abs(action)
        # moving cars
        nCar1 = min( nCar1 - action, MAX_CAR_AVAILABLE )
        nCar2 = min( nCar2 + action, MAX_CAR_AVAILABLE )

        sucs = []
        p_sum = 0
        for nCar1_prime , nCar2_prime in itertools.product( range( self.V.shape[0]), range( self.V.shape[1]  ) ):
            p, r = self.reward( (nCar1, nCar2), ( nCar1_prime , nCar2_prime ) )
            if p < 0.000001 : continue
            p_sum += p
            sucs.append( [p, r0+r , ( nCar1_prime , nCar2_prime ) ] )

        if p_sum < 0.5:
            for p,r,s_prime in sucs:
                print(p,r,s_prime)

        assert p_sum >= 0.5 , "{},suc total prob:{}".format( state, p_sum )

        # if p_sum != 0:
        #     for suc in sucs:
        #         suc[0] /= p_sum

        # print(suc)
        return sucs

    # probability in loc1 , if nCar delta = ?
    @functools.lru_cache(maxsize=None)
    def probRewardLoc1Delta(self, delta, nCar ):
        nIter = MAX_CAR_AVAILABLE + 1
        # req arbitarily, but return shoud be restricted according to nCar
        p_delta = [ stats.poisson.pmf( i,3 )* stats.poisson.pmf( min(i,nCar) +delta,3 ) for i in range(nIter) ]         
        return sum(p_delta) , sum( [ p* min(i,nCar)*10 for i,p in enumerate(p_delta) ] )

    # probability in loc2 , if nCar delta = ?
    @functools.lru_cache(maxsize=None)
    def probRewardLoc2Delta(self, delta, nCar ):
        nIter = MAX_CAR_AVAILABLE + 1
        # req arbitarily, but return shoud be restricted according to nCar
        p_delta = [ stats.poisson.pmf( i,4 )* stats.poisson.pmf( min(i,nCar)+delta,2 ) for i in range(nIter) ]         
        return sum(p_delta) , sum( [ p* min(i,nCar)*10 for i,p in enumerate(p_delta) ] )


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


if __name__ == "__main__":
    nSample = 1000000
    requests1 = np.random.poisson(3, nSample)
    returns1 = np.random.poisson( 3, nSample)
    requests2 = np.random.poisson(4, nSample)
    returns2 = np.random.poisson( 2, nSample)
    
    # 借出的车，比归还的车多一辆
    cnt = 0
    r = 0
    nCar1, nCar2, action = 16,16,2  # 14,18

    for i in range(nSample):
        rent1 = min( requests1[i], nCar1-action )
        rent2 = min( requests2[i], nCar2+action )
        if requests1[i] <= 20 and requests1[i] == returns1[i]  and returns1[i] <= rent1 and \
            requests2[i]<= 20 and  requests2[i] == returns2[i] and returns2[i] <= rent2 :
            cnt += 1
            r += (rent1+rent2) * 10
    print( cnt / nSample, -2*abs(action) + r / cnt )

    problem = Problem()
    problem.Initialization()
    sucs = problem.Successor( (nCar1, nCar2), action )
    for p,r,s_prime in sucs:
        # print(p,r,s_prime)
        if s_prime == (14,18):
            print( p,r )
        pass
    print(len(sucs))

    cnt = 0
    r = 0
    nCar1, nCar2, action = 1,1,-1 # 2,0,

    for i in range(nSample):
        rent1 = min( requests1[i], nCar1-action )
        rent2 = min( requests2[i], nCar2+action )
        if requests1[i] <= 20 and requests1[i] == returns1[i] and returns1[i] <= rent1 and \
            requests2[i]<= 20 and requests2[i] == returns2[i] and returns2[i] <= rent2 :
            cnt += 1
            r += (rent1+rent2) * 10
    print( cnt / nSample, -2*abs(action) + r / cnt )

    problem = Problem()
    problem.Initialization()
    sucs = problem.Successor( (nCar1, nCar2), action )
    for p,r,s_prime in sucs:
        # print(p,r,s_prime)
        if s_prime == (2,0):
            print( p,r )
        pass
    print(len(sucs))


        

