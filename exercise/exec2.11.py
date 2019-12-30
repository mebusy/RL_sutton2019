import numpy as np

SCALE = 1
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

    def Softmax( self, f ):
        # instead: first shift the values of f so that the highest number is 0:
        f -= np.max(f) # f becomes [-666, -333, 0]
        return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer

class epsilonGreedy( Learning ):

    def chooseAction(self, step ):
        # it should NOT update anything when choose action
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
            A = self.Argmax( self.Q +  np.nan_to_num (self.c * np.sqrt( np.log(step) / self.N  ) ))
        else:
            A = self.Argmax( self.Q  )
        return A 
        
class Gradient( epsilonGreedy ):
    def init(self , a ):
        self.stepsize = a 
        self.Q_bar = np.zeros( NACTIONS )
        self.H = np.zeros( NACTIONS )

    def chooseAction(self, step ):
        self.distribution = self.Softmax( self.H )
        A = np.random.choice( NACTIONS , p = self.distribution  )
        return A

    def performAction(self, step, A ):
        R = np.random.normal( env.means[step][A] , 1.0 )
        # before update average reward Q_bar,  update H af first
        for a in xrange( NACTIONS ):
            if a == A:
                self.H[a] += self.stepsize * ( R - self.Q_bar[a] ) * ( 1-self.distribution[a] )
            else:
                self.H[a] += self.stepsize * ( R - self.Q_bar[a] ) * (  -self.distribution[a] )

        self.Q_bar[A] = self.Q_bar[A] + self.stepsize*( R - self.Q_bar[A] )
        return R


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = ENV( ) 
    # for i in xrange(10):
    #     env.step( True )

    fig, ax = plt.subplots()

    eLearn = epsilonGreedy( )
    x = np.linspace( 0.01 , 0.5 , 50 )
    y = [ eLearn.doEpsoid( env, i , 0.1 )  for i in x ]
    plt.plot( x,y, color="red" )
    plt.text( 0.1, np.max( y )   , 'e-greedy', color='red', fontsize=14 )
    
    ucb = UCB( )
    x = np.linspace( 0.1 , 5 , 50 )
    y = [ ucb.doEpsoid( env, 0.1 , 0.1, i )  for i in x ]
    plt.plot( x,y, color="blue" )
    plt.text( 1, np.max( y )   , 'UCB', color='blue', fontsize=14 )

    gradient = Gradient( )
    x = np.linspace( 0.01 , 0.5 , 50 )
    y = [ gradient.doEpsoid( env,  i )  for i in x ]
    plt.plot( x,y, color="green" )
    plt.text( 0.1, np.min(y)   , 'Gradient', color='green', fontsize=14 )

    plt.xscale('log') 

    plt.text(0.2, -0.10, 'e', color='red', transform= ax.transAxes , fontsize=16 )
    plt.text(0.5, -0.10, 'a', color='green', transform=ax.transAxes, fontsize=16)
    plt.text(0.8, -0.10, 'c', color='blue', transform=ax.transAxes, fontsize=16)

    plt.show() 
    print "done"




