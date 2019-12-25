import numpy as np

SCALE = 10
STEP = 200000  / SCALE
STEP_TO_CALC_AVERAGE = 100000 / SCALE

NACTIONS = 10

class ENV():
    def __init__( self, debug=False ) :
        self.means = np.zeros( ( STEP , NACTIONS ) )

        # random walk at each step
        for i in xrange( STEP ):
            if i==0:
                self.means[i] = np.ones( ( 1, NACTIONS )  )
            else:
                self.means[i] = self.means[i-1] +  np.random.normal( 0 , 0.01 , size = NACTIONS )
        if debug :
            print self.means , self.means[0]

    

class Learning(object) :
    def __init__(self) :
        pass

    def Argmax( self, b  ):
        # tie breaking
        return np.random.choice(np.where(b == b.max())[0])

class epsilonGreedy( Learning ):

    def chooseAction(self, step ):
        if np.random.random() <= self.epsilon :
            A = np.random.choice( NACTIONS  )
        else:
            A = self.Argmax( self.Q )
        return A 

    def performAction(self, step, A ):
        R = np.random.normal( env.means[step][A] , 1.0 )
        self.Q[A] = self.Q[A] + self.stepsize*( R - self.Q[A] )
        return R

    def init(self , e, a):
        self.epsilon = e 
        self.stepsize = a 
        self.Q = np.zeros( NACTIONS )

    def doEpsoid( self, env , *arg  ):
        self.env = env 
        self.init( *arg )

        averageR = 0.0
        for i in xrange( STEP ):
            A = self.chooseAction(i ) 
            r = self.performAction(i, A ) 

            if True or i >= STEP_TO_CALC_AVERAGE:
                averageR  += ( r - averageR) / ( i+ 1 )

            if i%1000 == 0 :
                print "step:", i
             
        return averageR
          
class UCB( epsilonGreedy ):
    def init(self , e, a, c ):
        super( UCB , self ).init( e, a )

        self.c = c 
        self.N = np.zeros( NACTIONS, dtype=int )
    
    def performAction(self, step, A ):
        R = super( UCB , self).performAction( step, A )
        self.N[A] = self.N[A] + 1
        return R

    def chooseAction(self, step ):
        if np.random.random() <= self.epsilon :
            A = np.random.choice( NACTIONS  )
        else:
            A = self.Argmax( self.Q +  np.nan_to_num (self.c * np.sqrt( np.log(step) / self.N  ) ))
        return A 


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = ENV( ) 
    # for i in xrange(10):
    #     env.step( True )

    # eLearn = epsilonGreedy( )
    # x = np.linspace( 0.01 , 0.5 , 50 )
    # y = [ eLearn.doEpsoid( env, i , 0.1 )  for i in x ]
    # plt.plot(x,y, color="red" )

    ucb = UCB( )
    x = np.linspace( 0.1 , 10 , 50 )
    y = [ ucb.doEpsoid( env, 0.1 , 0.1, i )  for i in x ]
    plt.plot(x,y, color="blue" )




    plt.show() 
    print "done"




