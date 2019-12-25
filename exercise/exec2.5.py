import numpy as np
import time

# np.random.seed(  9173 )


means = [ 0.2, -0.8, 1.6, 0.4, 1.4,  -1.4 , -0.2, -0.9, 0.8 , -0.5  ]
sigma = 1
epsilon = 0.1

nStep = 10000
nExperiment = 20 #00 


bMofified10Arm = True

means_equal = np.array([ 1.0 ] * len(means))
best_action_value = np.zeros( ( 2, nExperiment ) ) 

def bandit( A) :
    if bMofified10Arm:
        return  np.random.normal( means_equal[A] , sigma  )
    else:
        return  np.random.normal( means[A] , sigma  )

# for sample average
averageRewards0 = np.zeros( ( nExperiment, nStep) )

def Argmax( b  ):
    return np.random.choice(np.where(b == b.max())[0])

def sample_average_experiment( idx  ):
    global means_equal
    nActions = len(means )

    Q = np.zeros( nActions )
    N = np.zeros( nActions, dtype=int )

    total = 0.0
    if bMofified10Arm:
        means_equal.fill( 1.0)
    # print Q 
    for i in xrange( nStep ) :
        if np.random.random() <= epsilon :
            A = np.random.choice( nActions  )
        else:
            A = Argmax( Q )

        R = bandit(A)
        N[A] = N[A] + 1
        Q[A] = Q[A] + (R - Q[A])/N[A] 
        
        total += R 
        averageRewards0[idx][i] = total/(i+1)

        if bMofified10Arm:
            means_equal += np.random.normal( loc=0.0, scale=0.01, size=10 ) 

    best_action_value[0][idx] = means_equal.max()


# for constant step 
averageRewards1 = np.zeros( ( nExperiment, nStep) )

def constant_step_experiment( idx  ):
    global means_equal
    nActions = len(means )

    Q = np.zeros( nActions )
    N = np.zeros( nActions, dtype=int )

    total = 0.0
    alpha = 0.1
    if bMofified10Arm:
        means_equal.fill( 1.0)
    # print Q 
    for i in xrange( nStep ) :
        if np.random.random() <= epsilon :
            A = np.random.choice( nActions  )
        else:
            A = Argmax( Q )

        R = bandit(A)
        N[A] = N[A] + 1

        # Q[A] = Q[A] + (R - Q[A])/N[A] 
        # the only change for a constant step 
        Q[A] = Q[A] + alpha*( R - Q[A] )

        total += R 
        averageRewards1[idx][i] = total/(i+1)

        if bMofified10Arm:
            means_equal += np.random.normal( loc=0.0, scale=0.01, size=10 ) 

    best_action_value[1][idx] = means_equal.max()


if __name__ == "__main__" :
    print max( means )

    for i in xrange( nExperiment ):
        sample_average_experiment(i)
    for i in xrange( nExperiment ):
        constant_step_experiment(i)

    best_q = best_action_value.mean(1)


    averageReward4eachStep0 = averageRewards0.mean( 0 )
    averageReward4eachStep1 = averageRewards1.mean( 0 )
    # print averageReward4eachStep
    import matplotlib.pyplot as plt
    plt.plot( xrange( nStep ) ,    averageReward4eachStep0 )
    plt.plot( xrange( nStep ) ,    averageReward4eachStep1 )

    plt.text( 4000,  best_q[1] * 0.4 ,r'average sample best:{}'.format( best_q[0] ) ,  color='blue' )
    plt.text( 4000,  best_q[1] * 0.35 ,r'const step-size best:{}'.format( best_q[1] ) ,  color='green' )
    plt.show()



    print "done"
