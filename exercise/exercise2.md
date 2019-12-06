
# Exercise of Chapter 2

- Exercise 2.1 
    - Q: In "ε-greedy action selection, for the case of two actions and ε = 0.5, what is the probability that the greedy action is selected?
    - A: *`(1-0.5) + (0.5/N)  > 0.5`*,  where N is the length of action space.

- Exercise 2.2  *Bandit example*
    - Q: Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using ε-greedy action selection, sample-average action-value estimates, and initial estimates of Q₁(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A₁ = 1, R₁ =-1,A₂ =2,R₂ =1,A₃ =2,R₃ =-2,A₄ =2,R₄ =2,A₅ =3,R₅ =0. On some of these time steps the ε case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?
    - A: on time steps 4,5 , it definitely occurred, 
        - on time steps 1,2,3  possibly occurred

- Exercise 2.3 
    - Q: In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively.
    - A: The mothod with ε=0.01 will perform best.  
        - It will find the optimal action about 99.1% of the time.





