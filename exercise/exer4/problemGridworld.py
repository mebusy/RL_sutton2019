
import numpy as np
import functools
from collections import defaultdict

N_STATE = 16
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3

class Problem(object):
    def __init__(self):
        pass

    def Initialization(self):
        self.V = np.zeros( N_STATE )  # np array
        self.PI = defaultdict( lambda: ( ACTION_UP, ACTION_DOWN, ACTION_RIGHT, ACTION_LEFT ) )   # 
        self.Gamma = 1

        self.PI[0] = ()
        self.PI[15] = ()
        
        # V[s] should give state value
        # V[s] gives policy
        return self.V, self.PI, self.Gamma
        
    def AllStates(self):
        return range( N_STATE )

    def IsTerminal( self, state ):
        return state == 0 or state == 15

    def AvailableActions(self, state ):
        return ( ACTION_UP, ACTION_DOWN, ACTION_RIGHT, ACTION_LEFT )

    def GetPolicy(self, state ):
        return self.PI[ state ]

    # return (  ( prob, reward, state'   ),  )
    @functools.lru_cache(maxsize=None)
    def Successor(self, state, action ):
        y = state // 4
        x = state % 4 

        if action == ACTION_LEFT:
            x = max(0, x-1)
        elif action == ACTION_RIGHT:
            x = min(3 , x+1 )
        elif action == ACTION_UP:
            y = max(0, y-1)
        elif action == ACTION_DOWN:
            y = min(3 , y+1 )

        s_prime = y*4 + x 
        return ( ( 1.0, -1, s_prime ), ) 

    def Show_cli(self):
        out = []
        for y in range(4):
            for x in range(4):
                s = y*4 + x
                val = self.V[s]
                out.append(  "%5.1f" % val  )
            out.append('\n')

        out = ''.join(out)
        print(out)

        out = []
        for y in range(4):
            for x in range(4):
                arrow = ""
                
                s = y*4 + x
                actions = self.PI[s]
                # assert len(actions) <= 2
                if ACTION_LEFT in actions:
                    arrow += "←"
                if ACTION_UP in actions:
                    arrow += "↑"
                if ACTION_DOWN in actions:
                    arrow += "↓"
                if ACTION_RIGHT in actions:
                    arrow += "→"

                out.append(  "  {}  ".format( arrow )  )
            out.append('\n')

        out = ''.join(out)
        print(out)




        

