# PROXIMAL POLICY OPTIMIZATION ALGORITHMS

------

## Proximal Policy Optimization


### TRPO vs PPO


PPO와 TRPO를 우선 비교해보자. 주어진 데이터를 가지고 현재 Policy를 최대한 큰 step만큼 빠르게 향상시키면서,그렇다고 성능이 발산해버릴 정도로 너무 큰 step으로 업데이트 하는 것을 억제하고자 한다.


TRPO와 PPO의 최적화식을 비교해보면 다음과 같다.


