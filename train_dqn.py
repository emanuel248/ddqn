from qnet import Q_Net
from environment import QEnvironment
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
import string
from random import randint,choice
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train_dqn(env):
    """
    :param env: environment object
    :type env: QEnvironment
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q_Net(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_target = copy.deepcopy(Q)

    Q.to(device)
    Q_target.to(device)
    
    optimizer = optim.Adam(Q.parameters(), lr=3e-4)

    epoch_num = 601
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-4
    epsilon_min = 0.1
    start_reduce_epsilon = 100
    train_freq = 10
    update_q_freq = 200
    gamma = 0.97
    show_log_freq = 5

    allchar = string.ascii_letters + string.digits
    proj_name = "".join(choice(allchar) for x in range(6))

    writer = SummaryWriter('logs/dqn_{proj_name}', flush_secs=20)

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        prev_obs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # select act
            prev_act = np.random.randint(3)
            if np.random.rand() > epsilon:
                prev_act = Q(torch.tensor(prev_obs, dtype=torch.float32).reshape(1, -1).to(device))
                prev_act = torch.argmax(prev_act.data)

            # act
            obs, reward, done = env.step(prev_act)

            # add memory
            memory.append((prev_obs, prev_act, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_prev_obs = torch.tensor(batch[:, 0].tolist(), dtype=torch.float32).reshape(batch_size, -1).to(device)
                        b_prev_act = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = torch.tensor(batch[:, 3].tolist(), dtype=torch.float32).reshape(batch_size, -1).to(device)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_prev_obs)
                        maxq = np.max(Q_target(b_obs).data, axis=1)
                        target = copy.deepcopy(q.data)
                        
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                        
                        # zero gradients before new backprop
                        Q.reset()
                        print(q,target)
                        loss = F.mse_loss(q, target)
                        total_loss += loss.data
                        loss.backward()
                        optimizer.step()

                if total_step % update_q_freq == 0:
                    Q_target = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            prev_obs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss.item())

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            writer.add_scalar('Loss', log_loss, epoch+1)
            writer.add_scalar('Reward', log_reward, epoch+1)
            print(' | '.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
    writer.close()
    return Q, total_losses, total_rewards
