import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        # self.fc1.weight.data.fill_(1)
        # self.fc1.bias.data.fill_(1)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc2.weight.data.fill_(1)
        # self.fc2.bias.data.fill_(1)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # self.fc3.weight.data.fill_(1)
        # self.fc3.bias.data.fill_(1)
        self.bn3 = nn.BatchNorm1d(self.n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_space):
        action_space = torch.tensor(action_space).to(self.device)
        state = state.to(self.device)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # print(x)
        # print(action_space)
        actions = torch.mul(torch.add(x, 0.0001), action_space)
        # print(actions)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, max_mem_size=int(3e04), eps_end=0.01,
                 eps_dec=2e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr=self.lr, n_actions=4, input_dims=input_dims, fc1_dims=128, fc2_dims=128)

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.action_space_memory = np.zeros((self.mem_size, 4), dtype=np.int32)
        self.new_action_space_memory = np.zeros((self.mem_size, 4), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, action_space, action_space_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.action_space_memory[index] = action_space
        self.new_action_space_memory[index] = action_space_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation, action_space):
        if np.random.random() > self.epsilon:
            # print(np.shape(observation))
            state = torch.tensor([observation]).to(self.Q_eval.device)
            state = state.float()
            actions = self.Q_eval.forward(state, action_space)
            action = torch.argmax(actions).item()
            if action_space == [0, 0, 0, 0]:
                action = -1
        else:
            if action_space == [0, 0, 0, 0]:
                action = -1
            else:
                recon = []
                for i in range(4):
                    if action_space[i] != 0:
                        recon.append(i)
                action = recon[np.random.randint(0, len(recon))]
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]
        action_space_batch = self.action_space_memory[batch]
        new_action_space_batch = self.new_action_space_memory[batch]

        q_eval = self.Q_eval.forward(state_batch, action_space_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch, new_action_space_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def save(self, path=os.getcwd(), name='neural_network'):
        self.Q_eval.eval()
        torch.save(self.Q_eval.state_dict(), name+'.pth')
        return True

    def load(self, path=os.getcwd(), name='neural_network'):
        self.Q_eval.load_state_dict(torch.load(name+'.pth'), strict=False)
        self.Q_eval.to(self.Q_eval.device)
        self.Q_eval.eval()
