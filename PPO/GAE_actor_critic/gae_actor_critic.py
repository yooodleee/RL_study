# GAE를 사용한 actor-critic 알고리즘을 pendulum-v1 환경에 적용함
# more information https://hiddenbeginner.github.io/Deep-Reinforcement-Learnings/book/Chapter2/8-gae.html

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

from collections import deque
from torch.distributions import Normal
from tqdm import tqdm


# GAEActorCrtic 클래스 구현

class GAEActorCritic:
    def __init__(
        self,
        state_dimension,
        action_dimension,
        hidden_dimsions=(512,),
        batch_size=32,
        activation_fn=F.relu,
        policy_lr=0.0003,
        value_lr=0.0003,
        gamma=0.99,
        lamb_da=0.95,
    ):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy=MLPGaussianPolicy(state_dimension, action_dimension, hidden_dimsions, activation_fn).to(self.device)
        self.value=MLPStateValue(state_dimension, hidden_dimsions, activation_fn).to(self.device)
        self.gamma=gamma
        self.lamb_da=lamb_da
        self.batch_size=batch_size

        self.policy_optimizer=torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer=torch.optim.Adam(self.value.parameters(), lr=value_lr)

        self.buffer=RolloutBuffer()
    
    def learn(self):
        self.policy.train()
        self.value.train()
        state, action, reward, next_state, done=self.buffer.sample()
        state, action, reward, next_state, done=map(lambda x: x.to(self.device), [state, action, reward, next_state, done])

        # GAE 계산
        with torch.no_grad():
            delta=reward + (1 - done) * self.gamma * self.value(next_state) - self.value(state) # \delta_t 담은 배열
            advantage=torch.clone(delta)    # gae를 담은 배열
            ret=torch.clone(reward) # return을 담은 배열
            for t in reversed(random(len(reward) - 1)):
                advantage[t] += (1 - done[t]) * self.gamma * self.lamb_da * advantage[t + 1]
                ret[t] += (1 - done[t]) * self.gamma * ret[t + 1]
        
        # \pi(a|s) 로그 확률 값 계산하기
        mu, std=self.policy(state)
        m=Normal(mu, std)
        z=torch.atanh(torch.clamp(action, -1.0 + 1e-7, 1.0 - 1e-7))
        log_probs=m.log_prob(z)

        # 가치 네트워크 업데이트하기
        value=self.value(state)
        value_loss=F.mse_loss(value, ret)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 정책 네트워크 업데이트
        policy_loss= -(log_probs * advantage).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        result={'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}


class RolloutBuffer:
    def __init__(self):
        self.buffer=list()
    
    def store(self, transition):
        self.buffer.append(transition)
    
    def sample(self):
        state, action, reward, next_state, done=map(np.array(), zip(*self.buffer))
        self.buffer.clear()
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    @property
    def size(self):
        return len(self.buffer)


# 정책 네트워크 및 상태 가치 네트워크 표현
class MLPGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dimension,
        action_dimension,
        hidden_dimensions=(512,),
        activation_fn=F.relu,
    ):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer=nn.Linear(state_dimension, hidden_dimensions[0])
        self.hidden_layers=nn.ModuleList()
        for i in range(len(hidden_dimensions) - 1):
            hidden_layer=nn.Linear(hidden_dimensions[i], hidden_dimensions[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.mu_layer=nn.Linear(hidden_dimensions[-1], action_dimension)
        self.log_std_layer=nn.Linear(hidden_dimensions[-1], action_dimension)
        self.activation_fn=activation_fn
    
    def forward(self, x):
        x=self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x=self.activation_fn(hidden_layer(x))
        
        mu=self.mu_layer(x)
        log_std=torch.tanh(self.log_std_layer(x))

        return mu, log_std.exp()


class MLPStateValue(nn.Module):
    def __init__(
        self,
        state_dimension,
        hidden_dimensions=(512,),
        activation_fn=F.relu,
    ):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer=nn.Linear(state_dimension, hidden_dimensions[0])
        self.hidden_layers=nn.ModuleList()
        for i in range(len(hidden_dimensions) - 1):
            hidden_layer=nn.Linear(hidden_dimensions[i], hidden_dimensions[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer=nn.Linear(hidden_dimensions[-1], 1)
        self.activation_fn=activation_fn
    
    def forward(self, x):
        x=self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x=self.activation_fn(hidden_layer(x))
        x=self.output_layer(x)

        return x


# GAE actor-critic 에이전트 구현
class GAEActorCritic:
    def __init__(
        self,
        state_dimension,
        action_dimension,
        hidden_dimensions=(512,),
        batch_size=32,
        activation_fn=F.relu,
        policy_lr=0.0003,
        value_lr=0.0003,
        gamma=0.99,
        lamb_da=0.95,
    ):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy=MLPGaussianPolicy(state_dimension, action_dimension, hidden_dimensions, activation_fn).to(self.device)
        self.value=MLPGaussianPolicy(state_dimension, hidden_dimensions, activation_fn).to(self.device)
        self.gamma=gamma
        self.lamb_da=lamb_da
        self.batch_size=batch_size

        self.policy_optimizer=torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer=torch.optim.Adam(self.value.parameters(), lr=value_lr)

        self.buffer=RolloutBuffer()
    
    @torch.no_grad()
    def act(self, state, training=True):
        self.policy.train(training)

        state=torch.as_tensor(state, dtype=torch.float, device=self.device)
        mu, std=self.policy(state)
        z=torch.normal(mu, std) if training else mu
        action=torch.tanh(z)

        return action.cpu().numpy()
    
    def learn(self):
        self.policy.train()
        self.value.train()
        state, action, reward, next_state, done=self.buffer.sample()
        state, action, reward, next_state, done=map(lambda x: x.to(self.device), [state, action, reward, next_state, done])

        # GAE 계산
        with torch.no_grad():
            delta=reward (1 - done) * self.gamma * self.value(next_state) - self.value(state)    # delta_t 담은 배열
            advantage=torch.clone(delta)    # gae를 담을 배열
            ret=torch.clone(reward) # return을 담을 배열
            for t in reversed(range(len(reward) - 1)):
                advantage[t] += (1 - done[t]) * self.gamma * self.lamb_da * advantage[t + 1]
                ret[t] += (1 - done[t]) * self.gamma * ret[t + 1]
        
        # \pi(a|s) 로그 확률 값 계산하기
        mu, std=self.policy(state)
        m=Normal(mu, std)
        z=torch.atanh(torch.clamp(action, -1.0 + 1e-7, 1.0 + 1e-7))
        log_probs=m.log_prob(z).sum(dim=-1, keepdim=True)

        # 가치 네트워크 업데이트하기
        value=self.value(state)
        value_loss=F.mse_loss(value, ret)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 정책 네트워크 업데이트
        policy_loss= -(log_probs * advantage).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        result={'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}

        return result
    
    def step(self, transition):
        result=None
        self.buffer.store(transition)
        if self.buffer.size >= self.batch_size:
            result=self.learn()
        
        return result


# ================================== Training ================================================
# GAE actor-critic 에이전트 훈련

def evaluate(env_name, agent, seed, eval_iterations):
    env=gym.make(env_name)
    scores=[]
    for i in range(eval_iterations):
        (state,_), terminated, truncated, score=env.reset(seed=seed + 100 + i), False, False, 0
        while not (terminated or truncated):
            a=agent.act(state, training=False)
            next_state, reward, terminated, truncated, _=env.step(2.0 * a)
            score += reward
            state=next_state
        scores.append(score)
    env.close()
    return round(np.mean(scores), 4)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


env_name='Pendulum-v1'

seed=1
seed_all(seed)
hidden_dimensions=(64, 64, )
max_iterations=1000000
eval_itervals=10000
eval_iterations=10
batch_size=64
activation_fn=torch.tanh
gamma=0.90
lamb_da=0.7

env=gym.make(env_name)
state_dimension=env.observation_space.shape[0]
action_dimension=env.action_space.shape[0]
agent=GAEActorCritic(
    state_dimension,
    action_dimension,
    hidden_dimensions=hidden_dimensions,
    activation_fn=activation_fn,
    batch_size=batch_size,
    gamma=gamma,
    lamb_da=lamb_da,
)

logger=[]
(state, _), terminated, truncated=env.reset(seed=seed), False, False
for t in tqdm(range(1, max_iterations + 1)):
    a=agent.act(state)
    next_state, reward, terminated, truncated, _=env.step(2.0 * a)
    result=agent.step((state, a, reward, next_state, terminated))
    state=next_state

    if result is not None:
        logger.append([t, 'policy_loss', result['policy_loss']])
        logger.append([t, 'value_loss', result['value_loss']])
    
    if terminated or truncated:
        (s, _), terminated, truncated=env.reset(), False, False
    
    if t % eval_itervals == 0:
        score=evaluate(env_name, agent, seed, eval_iterations)
        logger.append([t, 'Avg return', score])


logger=pd.DataFrame(logger)
logger.columns=['step', 'key', 'value']

fig=plt.figure(figsize=(12, 4))

ax=fig.add_subplot(1, 3, 1)
key='Avg return'
ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
ax.grid(axis='y')
ax.set_title("Average return over 10 episodes")
ax.set_xlabel("Step")
ax.set_ylabel("Avg return")

ax=fig.add_subplot(1, 3, 2)
key='policy_loss'
ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
ax.grid(axis='y')
ax.set_title("Policy loss")
ax.set_xlabel("Step")
ax.set_ylabel("Policy loss")

ax=fig.add_subplot(1, 3, 3)
key='value_loss'
ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
ax.grid(axis='y')
ax.set_title("value_loss")
ax.set_xlabel("step")
ax.set_ylabel("value_loss")

fig.tight_layout()
plt.show()