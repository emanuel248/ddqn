from qnet import Q_Net, Q_Net_RNN
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
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='tsla.us.txt', help='*.csv file path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--showplot', type=bool, default=False, help='plot training stats')
    parser.add_argument('--outdir', type=str, default='saved', help='model save dir')
    parser.add_argument('--weights', type=str, help='*.weights file path')
    opt = parser.parse_args()
    """
    :param env: environment object
    :type env: QEnvironment
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'using {device}')

    test,train = load_data(opt.data)
    env = QEnvironment(train, hist_t=20)
    #find out input size
    _inp = env.reset()
    actions = ['hold', 'long', 'sell', 'short', 'close']
    Q = Q_Net_RNN(device, input_size=_inp.shape[0]*_inp.shape[1], output_size=len(actions))
    Q_target = Q_Net_RNN(device, input_size=_inp.shape[0]*_inp.shape[1], output_size=len(actions))

    Q.to(device)
    Q_target.to(device)
    
    optimizer = optim.Adam(Q.parameters(), lr=0.3e-4)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.001, patience=25)

    epoch_num = opt.epochs
    step_max = env.len()
    print('env len', step_max)
    memory_size = 30000 #200 works
    batch_size = 1000 #100 works
    epsilon = 1.0
    epsilon_decrease = -0.001
    epsilon_min = 0.1
    start_reduce_epsilon = 50
    train_freq = 100
    update_q_freq = 20 #200 works
    gamma = 1.0
    show_log_freq = 1
    show_plot_freq = 20

    allchar = string.ascii_letters + string.digits
    proj_name = "".join(choice(allchar) for x in range(6))

    writer = SummaryWriter(f'logs/ddqn_{proj_name}', flush_secs=20)

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    with tqdm(total=100, position=1, bar_format='{desc}', desc='Stats') as desc:
        for epoch in tqdm(range(epoch_num), ncols=80, ascii=True):
            prev_obs = env.reset()

            step = 0
            done = False
            total_reward = 0
            total_loss = torch.tensor([0.0], dtype=torch.float32)

            for _ in range(step_max):
                if done:
                    break

                obs_ = env.current_obs()[0]
                if np.random.random() > epsilon:
                    prev_act = np.random.randint(0,5,1)
                else:
                    prev_act = Q(torch.tensor(prev_obs, dtype=torch.float32).reshape(1, obs_.shape[0], -1).to(device))
                    prev_act = torch.argmax(prev_act).item()
                
                
                #if len(env.positions) > 0 and prev_act != 2:
                #    prev_act = 0 #hold
                #if len(env.short_positions) > 0 and prev_act != 4:
                #    prev_act = 0 #hold
                
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
                            b_prev_obs = torch.tensor(batch[:, 0].tolist(), dtype=torch.float32).reshape(batch_size, obs_.shape[0], -1).to(device)
                            b_prev_act = np.array(batch[:, 1].tolist(), dtype=np.int32)
                            b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                            b_obs = torch.tensor(batch[:, 3].tolist(), dtype=torch.float32).reshape(batch_size, obs_.shape[0], -1).to(device)
                            b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                            q_raw = Q(b_prev_obs)
                            q = torch.argmax(q_raw, dim=1)

                            maxqs_raw = Q_target(b_obs)
                            maxqs = maxqs_raw.data

                            target = copy.deepcopy(q_raw.data)
                            #target = torch.zeros(q_raw.shape)
                            
                            for j in range(batch_size):
                                target[j, b_prev_act[j]] = b_reward[j]+gamma*maxqs[j, q[j]]
                            # zero gradients before new backprop
                            Q.reset()
                            #Q.reset_hidden()
                            loss = F.mse_loss(q_raw, target)
                            
                            total_loss += loss.data
                            loss.backward()
                            optimizer.step()

                    

                # next step
                total_reward += reward
                prev_obs = obs
                step += 1
                total_step += 1
                
            if epoch % update_q_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            total_rewards.append(total_reward)
            total_losses.append(total_loss.item())

            if epoch > epsilon_decrease and epsilon > epsilon_min:
                epsilon += epsilon_decrease
            if (epoch+1) % show_log_freq == 0:
                log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
                log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
                elapsed_time = time.time()-start
                writer.add_scalar('Loss', log_loss, epoch+1)
                writer.add_scalar('Reward', log_reward, epoch+1)

                desc.set_description(' | '.join(map(str, [epoch+1, f'eps {epsilon:.6f}, step {total_step}', f'reward {log_reward:.6f}', f'loss {log_loss:.6f}', f'value {env.balance:.3f}'])))
            if (epoch+1) % show_plot_freq == 0 and opt.showplot:
                plt.figure(figsize=(15,8))
                plt.subplot(3,1,1)
                sns.lineplot(x=range(env.data_combined.shape[0]), y=env.data_combined[:,5], label="long", linewidth=0.8)
                sns.lineplot(x=range(env.data_combined.shape[0]), y=env.data_combined[:,6], label="short", linewidth=0.8)
                plt.subplot(3,1,2)
                sns.lineplot(x=range(env.data_combined.shape[0]), y=env.data_combined[:,1], label="price", linewidth=0.8)
                plt.subplot(3,1,3)
                m = [n[2] for n in memory]
                sns.lineplot(x=range(len(m)), y=np.array(m), label="reward", linewidth=0.8)
                start = time.time()
                plt.show()
                #lr_schedule.step(log_loss) # worked without
    torch.save(Q.state_dict(), f'{opt.outdir}/model_{proj_name}.weights')
    writer.close()