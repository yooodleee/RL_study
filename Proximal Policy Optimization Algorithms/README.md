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


## 2 Background: Policy Optimization


### 2.1 Policy Gradient Methods


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


### Trust Region Methods


In TRPO, an objective function (the "surrogate" objective) is maximized subject to a constraint on the
size of the policy update. Specifically,


![image](https://github.com/user-attachments/assets/f648a023-8554-49ae-92f8-5ca1ba3e7bfa)


Here, θold is the vector of policy parameters before the update. This problem can efficiently be approximately solved using the conjugate gradient algorithm, after making a linear approximation to the objective and a quadratic approximation to the constraint.
The theory justifying TRPO actually suggests using a penalty instead of a constraint, i.e., solving the unconstrained optimization problem:


![image](https://github.com/user-attachments/assets/4b63b692-38bc-49d7-8f0a-d952abd36099)


for some coefficient β. This follows from the fact that a certain surrogate objective (which computes the max KL over state instead of the mean) forms a lower bound (i.e., a pessimistic bound) on the performance of the policy π. TRPO uses a hard constraint rather than a penalty because it is hard to choose a single value of β that performs well across different problems-or even within a single problem, where the the characteristics change over the course of learning. Hence, to achieve our goal of a first-order algorithm that emulates the monotonic improvement of TRPO, experiments show that is not sufficient to simply choose a fixed penalty coefficient β and optimize the penalized objective Equation (5) with SGD; additional modifications are required.


## 3 Clipped Surrogate Objective


Let rt(θ) denote the probability ratio rt(θ) = πθ(at | st) / πθold(at | st), so r(θold) = 1. TRPO maximizes a "surrogate" objective:


![image](https://github.com/user-attachments/assets/cb61e113-6235-43c9-afc6-3cd11bbab220)


The superscript CPI refers to conservative policy iteration, where this objective was proposed. Without a constraint, maximization of LCPI would lead to an excessively large policy update; hence, we now consider how to modify the objective, to penalize changes to the policy that move rt(θ) away from 1.
The main objective we propose we propose is the following:


![image](https://github.com/user-attachments/assets/c2ebace9-20cd-4d69-8412-2cb6e9292d89)


where epsilon is a hyperparameter, say, ε = 0.2. The motivation for this objective is as follows. The first term inside the min is LCPI. The second term, clip(rt(θ), 1-ε, 1+ε)A^t, modifies the surrogate objective by clipping the probability ratio, which removes the incentive for moving rt outside of the interval [1-ε, 1+ε]. Finally, we take the minimum of the clipped and unclipped objective, so the final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective. With this scheme, we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse. Note that LCLIP(θ)=LCPI(θ) to first order around θold (i.e., where r=1), however, they become different as θ moves away from θold. Figure 1 plots a single term (i.e., a single t) in LCLIP; note that the probability ratio r is clipped at 1-ε or 1+ε depending on whether the advantage is positive or negative.


![image](https://github.com/user-attachments/assets/21f594db-4595-4825-9bd6-621ce3f1fed9)
Figure 1: Plots showing one term (i.e., a single timestep) of the surrogate function LCLIP as a function of the probablity ratio r, for positive advantages (left) and negative advantages (right). The red circle on each plot shows the starting point for the optimizations, i.e., r=1. Note that LCLIP sums many of these terms.


Figure 2 provides another source of intuition about the surrogate objective LCLIP. It Shows how several objectives vary as we interpolate along the policy update direction, obtained by proximal policy optimization (the algorithm we will introduce shortly) on a continuous control problem. We can see that LCLIP is a lower bound on LCPI, with a penalty for having too large of a policy update.


![image](https://github.com/user-attachments/assets/e080a74f-f23f-4788-826b-ffc5e28f23d0)
Figure 2: Surrogate objectives, as we interpolate between the initial policy parameter θold, and the updated policy parameter, which we compute after one iteration of PPO. The updated policy has a KL divergence of about 0.02 from the initial policy, and this is the point at which LCLIP is maximal. This plot corresponds to the first policy update on the Hopper-v1 problem, using hyperparameters provided.


## 4 Adaptive KL Penalty Coefficient


Another approach, which can used as an alternative to the clipped surrogate objective, or in addition to it, is to use a penalty on KL divergence, and to adapt the penalty coefficient so that we acheive some target valeu of the KL divergence dtarg each policy update. In this experiments, the KL penalty performed worse than the clipped surrogate objective, however, researchers have included it here because it's an important baseline.


In the simplest instantiation of this algorithm, we perform the following steps in each policy update:


* Using severl epochs of minibatch SGD, optimize the KL-penalized objective :
![image](https://github.com/user-attachments/assets/ab22c3fd-3ec1-4718-b3aa-65d2544cc159)


* Computed d = = Eˆt[KL[πθold(· | st), πθ(· | st)]]
* If d < dtarg/1.5, β <- β/2
* If d > dtarg/1.5, β <- β*2


The updated β is used for the next policy update. With this scheme, we occasionally see policy updates where the KL divergence is significantly different from dtarg, however, these are rare, and β quickly adjust. The parameters 1.5 and 2 above are chosen heuristically, but the algorithm is not very sensitive to them. The initial value of β is a another hyperparmeter but is not important in practice because the algorithm quickly adjusts it.


## 5 Algorithm


The surrogate losses from the previous sections can be computed and differentiated with a minor change to a typical policy gradient implementation. For implementations that use automatic differentation, one simply constructs the loss LCLIP or LKLPEN instead of LPG, and one performs multiple steps of stochastic gradient ascent on this objective.


Most techniques for computing variance-reduced advantage-function estimators make use a learned state-value function V(s); for example, generalized advantage estimation, or the finite-horizon estimators in. If using a neural network architecture that shares parameters between the policy and value function, you must use a loss function that combines the policy surrogate and a value function error term. This objective can further be augmented by adding an entropy bouns to ensure sufficient exploration, as suggested in past work. Combining these terms, researchers obtain the following objective, which is (approximately) maximized each iteration:


![image](https://github.com/user-attachments/assets/73b64d6f-2849-41ff-a98c-090832a0b4be)


where c1, c2 are coefficient, and S denotes an entropy bouns, and LVF is a squared-error loss (V(st)-Vtarg)^2.
One style of policy gradient implementation, popularized in [Mni+16] and well-suited for use with recurrent neural networks, runs the policy for T timesteps (where T is much less than the episode length), and uses the collected samples for an update. This style requires an advantage estimator that does not look timestep T. The estimator used by [Mni+16] is:
![image](https://github.com/user-attachments/assets/e1d1469e-c0d9-4171-8a85-2c05ac8a9f80)


where t specifies the time index in [0, T], within a given length-T trajectorty segment. Generalizing this choice, we can use a truncated version of generalized advantage estimation, which reduces to Equation (10) when λ = 1:
![image](https://github.com/user-attachments/assets/a19c25df-5b76-4c39-b509-c0992f32e6df)


A proximal policy optimization (PPO) algorithm that uses fixed-length trajectort segements is shown below. Each iteration, each of N (parallel) actors collect T timesteps of data. Then we construct the surrogate loss on these NT timesteps of data, and optimize it with minibatch SGD (or usually for better performance, Adam), for K epochs.


![image](https://github.com/user-attachments/assets/056bb309-a1de-4190-885e-25e6d2275d57)


## 6 Experiments

## 6.1 Comparison of Surrogate Objective


First, we compare several 
