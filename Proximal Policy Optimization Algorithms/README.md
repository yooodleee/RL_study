# PROXIMAL POLICY OPTIMIZATION ALGORITHMS

---


## 1 Introduction


In recent years, several different approaches have been propsed for reinforcement learning with 
neural network function approximators. The leading contenders are deep Q-learning, "vanilla" 
policy gradient methods, and trust region / natural policy gradient methods. However, there is room
for improvement in developing a method that is scalable (to large models and parallel implementations),
data efficient, and robust (i.e., successful on a variety of problems without hyperparameter tuning).
Q-learning (with function approximation) fails on many simple problems and is poorly understood, 
vanilla policy gradient methods have poor data effiency and robustness; and trust region policy optimization
(TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).


This paper seeks to improve the current state of affairs by introducing an algorithm that attains the data efficency and reliable performance of TRPO, while using only first-order optimization.
This paper proposes a novel object with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy.To optimize policies, you must alternate between sampling data from the policy and performing several epochs of optimization on the sampled data.


This experiments compare the performance of various different versions of the surrogate objective, and find that the version with the clipped probability ratios performs best. On continuous control tasks, it performs better than the algorithms researchers compare against. On Atari, it performs significantly better (in terms of sample complexity) than A2C and similarly to ACER though it is much simpler.


## Background: Policy Optimization


### Policy Gradient Methods


policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent algorithm. The most commonly used gradient estimator has the form:


![image](https://github.com/user-attachments/assets/e9077a34-62ea-49a8-a9bf-10d5eb51c53b)

where πθ is a stochastic policy and A^t is an estimator of the advantage function at timestep t.
Here, the expectation Eˆt indicates the empirical average over a finite batch of samples, in an 
algorithm that algernates between sampling and optimization. Implementations that use automatic
differentiation software work by constructing an objective function whose gradient is the policy
gradient estimator; the estimator g^ is obtained by differentiating the objective:


![image](https://github.com/user-attachments/assets/abacfe8c-c2f8-4540-9f2c-b4603d10aefd)


While it is appealing to perform multiple steps of optimization on this loss LPG using the same 
trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy
updates (results are not shown but were similar or worse than the "no clipping or penalty" setting).


###Trust Region Methods


In TRPO, an objective function (the "surrogate" objective) is maximized subject to a constraint on the
size of the policy update. Specifically,


![image](https://github.com/user-attachments/assets/f648a023-8554-49ae-92f8-5ca1ba3e7bfa)


Here, θold is the vector of policy parameters before the update. This problem can efficiently be approximately solved using the conjugate gradient algorithm, after making a linear approximation to the objective and a quadratic approximation to the constraint.

