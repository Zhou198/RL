# Hands-on Projects of Reinforcement Learning

## Frozen lake: 
### keywords: Q-learning, $\varepsilon$-greedy policy, off-policy, valued-based RL

## CarPole:
### Actor-critic
It consists of actor/policy network and critic network, where the former is used to take actions while the later guilds the former's behavior. 
Specifically, the actor network is updated based on the policy gradient by minimizing the "weighted" cross-entropy (or maximizing the weighted log-likelihood) for the pair $\{(s_t, a_t)\}$. Here the "weight" is a value in critic.
There are multiple variants of such value:
1. Total discounted reward: $\sum_{t=0}^\infty\gamma^tr_t$.
2. Discounted reward after taking the action $a_t$: $\sum_{t=t'}^\infty\gamma^{t-t'}r_t$
3. Discounted reward centered by the baseline $b(s_t)$ (cannot depend on $a_t$): $\sum_{t=t'}^\infty\gamma^{t-t'}r_t- b(s_t)$.
4. State-action value: $Q(s_t, a_t)$.
5. Advantage Function: $Q(s_t, a_t)-V(s_t)$, which is similar to 3.
6. Temporal Difference (TD) error: $r_t+\gamma V(s_t)-V(s_{t+1})$.

Note that $r_t$'s in variant 1-3 are calculated by MC and finalized only when the current episode stops. This indicates the updating is time-consuming, and variance is large (sampling at each step), even though it is less biased (or even unbiased for variant 1).
Considering these limitations, methods involving estimating $Q(s_t, a_t)$ or $V(s_t)$ instead of purely sampling within each episode are introduced as in variant 4-5. The caveat is they are biased estimation. In contrast, variant 6 takes the trade-off between aforementioned two camps since it utilizes both one-step sampling and valued estimation. 

On the other hand, the critic network is to approximate the value function, which is updated by minimizing the MSE. 

To improve the performance, two tricks inspired from the DQN are further employed in actor-critic: buffer replay and the usage of fixed target network.

### Proximal Policy Optimization (PPO)

![carRace](./fig/rl0.gif)

*This is the caption for my animation.*

