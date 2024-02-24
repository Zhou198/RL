import numpy as np
import torch
import torch.nn.functional as F
import copy


class DQLearning:
    def __init__(self, params, net):
        self.params = params
        self.net_cur = net.to(params.device)
        self.net_trg = copy.deepcopy(self.net_cur)
        self.opt = torch.optim.Adam(self.net_cur.parameters(), lr=params.critic_lr, weight_decay=params.wd)
        self.net_loss = np.zeros(params.total_episodes)

    def update(self, states, actions, rewards, next_states, dones, episode, reload):
        if reload:
            self.net_trg.load_state_dict(self.net_cur.state_dict())

        self.net_cur.train()
        self.net_cur.zero_grad()
        self.net_trg.eval()

        with torch.no_grad():
            G_val = self.net_trg(next_states).max(-1)[0]
            true_Q = rewards + self.params.gamma * G_val * (1 - 1.0 * dones)

        pred_Q = self.net_cur(states).gather(1, actions.reshape(-1, 1)).reshape(-1, )
        net_loss_step = F.mse_loss(pred_Q, true_Q)

        net_loss_step.backward()
        self.opt.step()

        self.net_loss[episode] += net_loss_step.detach()

    def take_action(self, state):
        q_val = self.net_cur(state)
        max_val, action = q_val.max(-1)
        u = np.random.rand()
        if u < self.params.epsilon or torch.all(max_val[0] == q_val):
            action = torch.tensor(np.random.choice(self.params.action_size, 1)).to(self.params.device)

        return action


class ActorCritic:
    def __init__(self, params, net_actor, net_critic):
        self.params = params
        self.actor = net_actor.to(params.device)
        self.critic_cur = net_critic.to(params.device)
        self.critic_trg = copy.deepcopy(self.critic_cur)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=params.actor_lr, weight_decay=params.wd)
        self.opt_critic = torch.optim.Adam(self.critic_cur.parameters(), lr=params.critic_lr, weight_decay=params.wd)

        self.critic_net_loss = np.zeros(params.total_episodes)
        self.actor_net_loss = np.zeros(params.total_episodes)

    def update(self, states, actions, rewards, next_states, dones, episode, reload, **kwargs):
        if reload:
            self.critic_trg.load_state_dict(self.critic_cur.state_dict())

        self.actor.train()
        self.critic_cur.train()

        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()

        self.critic_trg.eval()

        with torch.no_grad():
            G_val = self.critic_trg(next_states).reshape(-1, )
            if self.params.value != "state": G_val = G_val.max(-1)[0]
            true_Q = rewards + self.params.gamma * G_val * (1 - 1.0 * dones)

        pred_Q = self.critic_cur(states).reshape(-1, )
        if self.params.value != "state": pred_Q = pred_Q[torch.arange(0, len(true_Q)), actions]
        critic_loss_step = F.mse_loss(pred_Q, true_Q)
        actor_loss_step = self.actor_loss(states, actions, true_Q - pred_Q.detach())

        critic_loss_step.backward()
        self.opt_critic.step()

        actor_loss_step.backward()
        self.opt_actor.step()

        self.critic_net_loss[episode] += critic_loss_step.detach()
        self.actor_net_loss[episode] += actor_loss_step.detach()

    def actor_loss(self, states, actions, td_err):
        action_prob = self.policy(states).gather(1, actions.reshape(-1, 1)).reshape(-1,)
        return torch.mean(-torch.log(action_prob) * td_err)

    def policy(self, state):
        logits = self.actor(state).clip(-15, 15)
        prob = F.softmax(logits, -1)
        return prob

    def take_action(self, state):
        return torch.multinomial(self.policy(state), 1)[0]


class PPO(ActorCritic):
    # def __init__(self, params, net_actor, net_critic):
    #     super(PPO, self).__init__(params, net_actor, net_critic)

    def update(self, states, actions, rewards, next_states, dones, episode, reload, **kwargs):
        if reload:
            self.critic_trg.load_state_dict(self.critic_cur.state_dict())

        self.actor.train()
        self.critic_cur.train()

        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()

        self.critic_trg.eval()

        with torch.no_grad():
            G_val = self.critic_trg(next_states).reshape(-1, )
            td_trg = rewards + self.params.gamma * G_val * (1 - 1.0 * dones)

        td_err = td_trg - self.critic_cur(states).reshape(-1, ).detach()
        gae = self.general_advantage_est(td_err)

        prev_prob = self.policy(states).gather(1, actions.reshape(-1, 1)).detach().reshape(-1,)

        for _ in range(self.params.epochs):
            td_pred = self.critic_cur(states).reshape(-1, )
            critic_loss_step = F.mse_loss(td_pred, td_trg)
            actor_loss_step = self.actor_loss(states, actions, prev_prob, gae)

            critic_loss_step.backward()
            self.opt_critic.step()

            actor_loss_step.backward()
            self.opt_actor.step()

            self.critic_net_loss[episode] += critic_loss_step.detach()
            self.actor_net_loss[episode] += actor_loss_step.detach()

    def actor_loss(self, states, actions, prev_prob, gae):
        cur_prob = self.policy(states).gather(1, actions.reshape(-1, 1)).reshape(-1,)
        weightIS = cur_prob / prev_prob

        loss_clip = torch.fmin(weightIS * gae, weightIS.clip(1 - self.params.clip_eps, 1 + self.params.clip_eps) * gae)
        return torch.mean(-loss_clip)

    def general_advantage_est(self, td_err):
        n = len(td_err)
        advtg = torch.zeros_like(td_err)
        gae = 0

        for i in range(n - 1, -1, -1):
            gae = td_err[i] + self.params.gamma * self.params.lam * gae
            advtg[i] = gae

        return gae



