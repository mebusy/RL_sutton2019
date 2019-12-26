
# S.Sutton  Reinforcement Learning: An Introduction

http://incompleteideas.net/book/

http://incompleteideas.net/book/solutions-2nd.html

Recommend :

- Covering Chapter 1 for a brief overview.
- Chapter 2 through Section 2.4, Chapter 3, and then selecting sections from the remaining chapters according to time and interests.
- Chapter 6 is the most important for the subject and for the rest of the book.
- artificial intelligence or planning :  Chapter 8
- machine learning or neural networks :  Chapter 9, 10

marked as `*` :  more difficult and not essential to the rest of the book.  These can be omitted on 1st reading. 

## Chapter 1. Introduction 

[Introduction](book/Introduction.pdf)

- key features
    - trial-and-error search 
    - delayed reward
    - considers the *whole* problem of a goal-directed agent interacting with an uncertain environment.
- trade-off between exploration and exploitation
- four main subelements
    1. policy
    2. reward signal
        - immediate desirability of state
        - sole objective of RL is to maximize the total reward
    3. value function
        - value indicate the long-term desirability of states
    4. (optionally) model of the environment.
        - model-based methods: use models and *planning*
        - model-free methods: trial-and-error
- play Tic-Tac-Toe against an imperfect player
    - minimax is not correct here
    - dynamic programming requires a complete model, need playing many games to build from experience (this is not that different from some of the reinforcement learning methods).
- reinforcement learning can also be applied 
    - when part of the state is hidden,
    - or to continuous-time problems as well, 
    - model is not required, but models can easily be used,
    - at both high and low levels in a system.
- The use of value functions distinguishes reinforcement learning methods from evolutionary methods.



# Part I: Tabular Solution Methods

[Part I](book/Part%20I%20Tabular%20Solution%20Methods/Part%20I.pdf)

The simplest RL forms: the approximate value functions to be represented as *arrays*, or *tables* , and can ofen find *exact solution*, while general RL can only find approximate solutions. 

1. bandit problem
    - only a single state
2. MDP ,  3 fundamental mothods to solve MDP:
    - dynamic programming
        - well developed mathematically
        - but require a complete and accurate model of the environment
    - Monte Carlo methods
        - don’t require a model and are conceptually simple
        - but are not well suited for step-by-step incremental computation.
    - temporal-difference learning 
        - require no model and are fully incremental
        - but are more complex to analyze.
3. combining those 3 methods to solve MDP

## Chapter 2. Multi-armed Bandits

[Chapter 2](book/Part%20I%20Tabular%20Solution%20Methods/02.%20Multi-armed%20Bandits.pdf)

- k-armed Bandit Problem
- balancing exploration and exploitation
    1. ε-greedy methods
        - choose actions randomly a small fraction of the time
            - tie breaking on greedy *argmax*
        - small ε improved more slowly, but eventually would perform better 
        - estimate action values
            - averaging methods (for stationary problem) ,  special const step-size method, with α = 1/n
            - const step-size method (for nonstationary problem)
    2. UCB
        - different on **exploration** 
        - not pratical in general RL problem.
    3. Gradient Bandit Algorithms
        - estimate not action values, but action preferences
        - learning a numerical *preference* for each action  H<sub>t</sub>(a).
        - [soft-max](https://github.com/mebusy/notes/blob/master/dev_notes/softmax.md) distribution
- optimistic initial values
    - encourage exploration, but is effective only on stationary problems
    - may use *Unbiased Constant-Step-Size Trick* to utilize initial value in nonstationary problem.


## Chapter









