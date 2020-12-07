
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


## Chapter 3. Finite Markov Decision Processes

[Chapter 3](book/Part%20I%20Tabular%20Solution%20Methods/03%20Finite%20MDP.pdf)

- 3.1 The Agent–Environment Interface
    - *Markov property* states that the conditional probability distribution for the system at the next step (and in fact at all future steps) 
        - depends ONLY on the current state of the system, and not additionally on the state of the system at previous steps.
- 3.2 Goals and Rewards
    - It is thus critical that the rewards we set up truly indicate what we want accomplished.
        - In particular, the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do.
        - For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent’s pieces or gaining control of the center of the board.
        - The reward signal is your way of communicating to the robot **what** you want it to achieve, not **how** you want it achieved
- 3.3 Returns and Episodes
    - *episode*: the agent–environment interaction breaks naturally into subsequences, which we call episodes.
    - *episodic tasks*: Tasks with episodes of this kind are called episodic tasks.
    - *continuing tasks*: In many cases the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit.
        - the final time step would be T = ∞, and the return could itself easily be infinite.
        - Solution: discounted rewards
    - Example: Pole-Balancing
        - This task could be treated as episodic. The reward in this case could be +1 for every time step on which failure did not occur.
        - Alternatively, we could treat pole-balancing as a continuing task, using discounting. In this case the reward would be -1 on each failure and zero at all other times.
- 3.5 Policies and Value Functions
    - state-value function / action-value function
    - The *value* of a *state* is the expected sum of all future rewards when starting in that **state** and following a specific policy.
        - **Note that the value of the terminal state, if any, is always zero.**
    - The value functions v<sub>π</sub> and q<sub>π</sub> can be estimated from experience.
        - We call estimation methods of this kind *Monte Carlo methods* because they involve averaging over many random samples of actual returns.
- 3.6 Optimal Policies and Optimal Value Functions
    - Explicitly solving the Bellman optimality equation is rarely directly useful. This solution relies on at least three assumptions that are rarely true in practice: 
        1. know the exact MDP
        2. have enough computational resources
        3. the Markov property
    - We consider a variety of such methods in the following chapters.


## Chapter 4. Dynamic Programming

[Chapter 4](book/Part%20I%20Tabular%20Solution%20Methods/04%20Dynamic%20Programming.pdf)

- Classical DP algorithms are of limited utility in reinforcement learning
    - both because of their assumption of a perfect model and because of their great computational expense.
- DP provides an essential foundation for the understanding of the methods presented in the rest of this book.
    - In fact, all of these methods can be viewed as attempts to achieve much the same effect as DP, only with less computation and without assuming a perfect model of the environment.
- We usually assume that the environment is a finite MDP, although DP ideas can be applied to problems with continuous state and action spaces, exact solutions are possible only in special cases. 
    - a common way is to quantize the state and action spaces and then apply finite-state DP methods
- Two most popular DP methods
    - 4.3 Policy Iteration
        - Policy Evaluation + Policy Improvement
        - *policy evaluation*: compute the state-value function v<sub>π</sub> for an arbitrary policy π. (until converge)
        - using the value function for a policy to help find better policies.
    - 4.4 Value Iteration

## Chapter 5 Monte Carlo Methods

[Chapter 5](book/Part%20I%20Tabular%20Solution%20Methods/05%20Monte%20Carlo%20Methods.pdf)

- **estimating** value functions, **not assume** complete knowledge of the environment, based on averaging **complete** returns (as opposed to methods that learn from partial returns).
- Monte Carlo methods require only *experience*
    - Learning from *actual* experience requires no prior knowledge of the environment’s dynamics.
    - Learning from *simulated* experience, a model is required, the model need only generate sample transitions, not the complete probability distributions.
        - e.g. Blackjack. 
- 5.1 Prediction
    - *first-visit* MC method  /  **every-visit** MC method
- 5.2 Monte Carlo Estimation of Action Values
    - If a model is not available, state-value alone is not sufficient to determine a policy.
    - The only complication is that many state–action pairs may never be visited. For policy evaluation to work for action values, we must assure continual exploration.
- 5.3 Monte Carlo Control
    - Monte Carlo ES(Exploring Starts) , After each episode , do evaluation and improvement
- 5.4 Monte Carlo Control without Exploring Starts
    - **On-policy first-visit MC control** (for ε-soft policies)
- 5.5 Off-policy Prediction via Importance Sampling
- Advantage than DP
    1. with no model of the environment’s dynamic
    2. can be used with simulation or *sample models*
    3. the estimates for each state are independent
        - can evaluate a single state without forming estimates for any other states. 

<details>
<summary>
<b>On-policy VS Off-policy</b>
</summary>

- off-policy: learning is from data **off** the target policy
- [强化学习中on-policy 与off-policy有什么区别](https://www.zhihu.com/question/57159315)
    - 皇帝希望能多了解民间百姓的生活
        - on-policy 微服出巡
        - off-policy 派多个官员去了解情况
- The most important difference is how Q is updated after each action.
    - ![](imgs/RL-on-off-policy.jpg)
    - SARSA uses the Q' following a ε-greedy policy exactly as A' is drawn from it.
    - Q-learning uses the maximum Q' over all possible actions for the next step. This makes it look like following a greedy policy.

· | SARSA | Q-learning
--- | --- | --- 
choosing A' | π | π
updating Q | π | μ

where π is ε-greedy policy , and μ is a greedy policy

</details>

## Chapter 6 Temporal-Difference Learning

[Chapter 6](book/Part%20I%20Tabular%20Solution%20Methods/06%20TD-Learning.pdf)

- TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas
    - can learn directly from raw experience without a model
    - update estimates based on other learned estimates
- 6.1 TD Prediction
    - *bootstrapping* method
        - update base on existing estimate
    - TD(0), or *one-step* TD, update state-value only until the next time step
- 6.2 Advantages of TD Prediction Methods
    - do not require a model of the environment, of its R and P  ( over DP)
    - are naturally implemented in an online, fully incremental fashion ( over MC )
        - some applications have very long episodes, other applications are continuing tasks and have no episodes at all
- 6.4 Sarsa: On-policy TD Control
- 6.5 Q-learning: Off-policy TD Control
- 6.6 Expected Sarsa
    - when updating Q value, use expected value instead of max ( compare with Q-learning )
- 6.7 Maximization Bias and Double Learning

<details>
<summary>
<b>When are Monte Carlo methods preferred over temporal difference ones?</b>
</summary>

- main difference
    - TD-learning uses bootstrapping to approximate the action-value function
    - Monte Carlo uses an average to accomplish this
- The main problem with TD learning and DP is that their step updates are biased on the initial conditions of the learning parameters. 
    - However, the bias can cause significant problems, especially for off-policy methods (e.g. Q Learning) and when using function approximators. That combination is so likely to fail to converge that it is called the **deadly triad** in Sutton & Barto.
- Monte Carlo control methods do not suffer from this bias, as each update is made using a true sample of what Q(s,a) should be. However, Monte Carlo methods can suffer from high variance, which means more samples are required to achieve the same degree of learning compared to TD.
- If you are using a value-based method (as opposed to a policy-based one), then TD learning is generally used more in practice, or a TD/MC combination method such as TD(λ) can be even better. 
    - Monte Carlo learning is conceptually simple, robust and easy to implement. I would generally not use it for a learning controller engine (unless in a hurry to implement something for a simple environment), but I would seriously consider it for policy evaluation in order to compare multiple agents for instance.


</details>



