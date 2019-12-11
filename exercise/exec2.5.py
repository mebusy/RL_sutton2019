import numpy as np
import time

means = [ 0.2, -0.8, 1.6, 0.4, 1.4,  -1.4 , -0.2, -0.9, 0.8 , -0.5  ]
sigma = 0.6
epsilon = 0.1

nStep = 10000
nExperiment = 20 #00 


def bandit( A) :
    return np.random.normal( means[A] , sigma  )

# for sample average
averageRewards0 = np.zeros( ( nExperiment, nStep) )

def sample_average_experiment( idx  ):
    nActions = len(means )
    Q = np.zeros( nActions )
    N = np.zeros( nActions, dtype=int )

    total = 0.0
    # print Q 
    for i in xrange( nStep ) :
        if np.random.random() <= epsilon :
            A = np.random.choice( nActions  )
        else:
            A = np.argmax( Q )

        R = bandit(A)
        N[A] = N[A] + 1
        Q[A] = Q[A] + (R - Q[A])/N[A] 
        
        # 1. make Nonstationary 
        Q[A] += np.random.normal( 0, 0.01 )
        
        total += R 
        averageRewards0[idx][i] = total/(i+1)


# for constant step 
averageRewards1 = np.zeros( ( nExperiment, nStep) )

def constant_step_experiment( idx  ):
    nActions = len(means )
    Q = np.zeros( nActions )
    N = np.zeros( nActions, dtype=int )

    total = 0.0
    alpha = 0.1
    # print Q 
    for i in xrange( nStep ) :
        if np.random.random() <= epsilon :
            A = np.random.choice( nActions  )
        else:
            A = np.argmax( Q )

        R = bandit(A)
        N[A] = N[A] + 1

        # Q[A] = Q[A] + (R - Q[A])/N[A] 
        # 2. the only change for a constant step 
        Q[A] = Q[A] + alpha*( R - Q[A] )

        # 1. make Nonstationary 
        Q[A] += np.random.normal( 0, 0.01 )
        
        total += R 
        averageRewards1[idx][i] = total/(i+1)



if __name__ == "__main__" :
    print max( means )
    for i in xrange( nExperiment ):
        sample_average_experiment(i)
    for i in xrange( nExperiment ):
        constant_step_experiment(i)

    averageReward4eachStep0 = averageRewards0.mean( 0 )
    averageReward4eachStep1 = averageRewards1.mean( 0 )
    # print averageReward4eachStep
    import matplotlib.pyplot as plt
    plt.plot( xrange( nStep ) ,    averageReward4eachStep0 )
    plt.plot( xrange( nStep ) ,    averageReward4eachStep1 )
    plt.show()

    print "done"
