import gym
import itertools
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque, namedtuple

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *

from model import *
from funcs import *


#----------------------------------------------------------------------------------

ALGO = "DDQN" #"DQN"  # DDQN

VALID_ACTIONS = [0, 1, 2] # LEFT, RIGHT, ATTACK

#----------------------------------------------------------------------------------

# set parameters for running

train_or_test = 'train' #'test' #'train'
train_from_scratch = True
start_iter = 0
start_episode = 0
epsilon_start = 1.0

#----------------------------------------------------------------------------------

game = doom_game()

game.new_episode()
game_state = game.get_state()
game_var = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]

action_size = game.get_available_buttons_size()
print("action_size: ", action_size) # LEFT, RIGHT, ATTACK

print("Buttons: ", game.get_available_buttons())

print("episode timeout: ", game.get_episode_timeout())


state_img = game_state.screen_buffer # new_state  
state_img = np.rollaxis(state_img,0,3)
plt.imsave("Doom1.png", state_img)

#----------------------------------------------------------------------------------


# experiment dir
experiment_dir = os.path.abspath("./experiments/VizDoom")


# create ckpt directory    
checkpoint_dir = os.path.join(experiment_dir, "ckpt")
checkpoint_path = os.path.join(checkpoint_dir, "model")
    
if not os.path.exists(checkpoint_dir):
   os.makedirs(checkpoint_dir)

#----------------------------------------------------------------------------------





def deep_q_learning(sess, game, q_net, target_net, state_processor, num_episodes, train_or_test='train', train_from_scratch=True,
                    start_iter=0, start_episode=0, replay_memory_size=250000, replay_memory_init_size=50000, update_target_net_every=10000,
                    gamma=0.99, epsilon_start=1.0, epsilon_end=[0.1,0.01], epsilon_decay_steps=[1e5,1e5], batch_size=32):
                   
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # policy 
    policy = epsilon_greedy_policy(q_net, len(VALID_ACTIONS))


    # populate replay memory
    if (train_or_test == 'train'):
      print("populating replay memory")
      replay_memory = populate_replay_mem(sess, game, state_processor, replay_memory_init_size, policy, epsilon_start, 
                                                       epsilon_end[0], epsilon_decay_steps[0], VALID_ACTIONS, Transition)


    # epsilon start
    if (train_or_test == 'train'):
       delta_epsilon1 = (epsilon_start - epsilon_end[0])/float(epsilon_decay_steps[0])     
       delta_epsilon2 = (epsilon_end[0] - epsilon_end[1])/float(epsilon_decay_steps[1])    
       if (train_from_scratch == True):
          epsilon = epsilon_start
       else:
          if (start_iter <= epsilon_decay_steps[0]):
             epsilon = max(epsilon_start - float(start_iter) * delta_epsilon1, epsilon_end[0])
          elif (start_iter > epsilon_decay_steps[0] and start_iter < epsilon_decay_steps[0]+epsilon_decay_steps[1]):
             epsilon = max(epsilon_end[0] - float(start_iter) * delta_epsilon2, epsilon_end[1])
          else:
             epsilon = epsilon_end[1]      
    elif (train_or_test == 'test'):
       epsilon = epsilon_end[1]


    # total number of time steps 
    total_t = start_iter


    for ep in range(start_episode, num_episodes):

        # save ckpt
        saver.save(tf.get_default_session(), checkpoint_path)

        # game reset
        game.new_episode()
        game_state = game.get_state()
        game_var = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        
        state_img = game_state.screen_buffer # new_state  
        state_img = np.rollaxis(state_img,0,3)   

        state = state_processor.process(sess, state_img)
        state = np.stack([state] * 4, axis=2)

        loss = 0.0
        time_steps = 0
        episode_rewards = 0.0
    


        while True:
            
            if (train_or_test == 'train'):
              #epsilon = max(epsilon - delta_epsilon, epsilon_end) 
              if (total_t <= epsilon_decay_steps[0]):
                    epsilon = max(epsilon - delta_epsilon1, epsilon_end[0]) 
              elif (total_t >= epsilon_decay_steps[0] and total_t <= epsilon_decay_steps[0]+epsilon_decay_steps[1]):
                    epsilon = epsilon_end[0] - (epsilon_end[0]-epsilon_end[1]) / float(epsilon_decay_steps[1]) * float(total_t-epsilon_decay_steps[0]) 
                    epsilon = max(epsilon, epsilon_end[1])           
              else:
                    epsilon = epsilon_end[1]


              # update target net
              if total_t % update_target_net_every == 0:
                 copy_model_parameters(sess, q_net, target_net)
                 print("\n copied params from Q net to target net ")

                   
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

#------             

            game.set_action(action_list(action))
            frame_per_action = 8   
            game.advance_action(frame_per_action)  # skiprate  
            reward = game.get_last_reward()


            done = game.is_episode_finished()
  
            if (not done):
              # non-Terminal state
              new_game_state = game.get_state()
              new_game_var = new_game_state.game_variables    
            else:
              # Terminal state
              new_game_state = game.get_state()
              new_game_var = game_var

            reward = shape_reward(reward, new_game_var, game_var)
  
#-------

            if (not done):
              next_state_img = new_game_state.screen_buffer # new_state  
              next_state_img = np.rollaxis(next_state_img,0,3) 
              next_state_img = state_processor.process(sess, next_state_img)
           
              # state is of size [84,84,4]; next_state_img is of size[84,84]
              next_state = np.zeros((84,84,4),dtype=np.uint8)
              next_state[:,:,0] = state[:,:,1] 
              next_state[:,:,1] = state[:,:,2]
              next_state[:,:,2] = state[:,:,3]
              next_state[:,:,3] = next_state_img    
            else:
              # Terminal state
              next_state = state 


            episode_rewards += reward  
            time_steps += 1

 
            if (train_or_test == 'train'):

                # if replay memory is full, pop the first element
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                # save transition to replay memory
                replay_memory.append(Transition(state, action, reward, next_state, done))   
                            

                # sample a minibatch from replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))



                # calculate q values and targets 

                if (ALGO == 'DQN'): 

                   q_values_next = target_net.predict(sess, next_states_batch)
                   greedy_q = np.amax(q_values_next, axis=1) 
                   targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * greedy_q

                elif (ALGO == 'DDQN'):
                 
                   q_values_next = q_net.predict(sess, next_states_batch)
                   greedy_q = np.argmax(q_values_next, axis=1)
                   q_values_next_target = target_net.predict(sess, next_states_batch)
                   targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * q_values_next_target[np.arange(batch_size), greedy_q]


                # update net 
                if (total_t % 4 == 0):
                   states_batch = np.array(states_batch)
                   loss = q_net.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            game_var = new_game_var  
            total_t += 1
            

        if (train_or_test == 'train'): 
           print('\n Eisode: ', ep, '| time steps: ', time_steps, '| total episode reward: ', episode_rewards, '| total_t: ', total_t, '| epsilon: ', epsilon, '| replay mem size: ', len(replay_memory))
        elif (train_or_test == 'test'):
           print('\n Eisode: ', ep, '| time steps: ', time_steps, '| total episode reward: ', episode_rewards, '| total_t: ', total_t, '| epsilon: ', epsilon)
        print("killcount: ", game_var[0], "ammo: ", game_var[1], "health: ", game_var[2]) 


        if (train_or_test == 'train'):
            f = open("experiments/VizDoom" + "/performance.txt", "a+")
            f.write(str(ep) + " " + str(time_steps) + " " + str(episode_rewards) + " " + str(total_t) + " " + str(epsilon) + " " + str(game_var[0]) + " " + str(game_var[1]) + " " + str(game_var[2]) + '\n')  
            f.close()

#----------------------------------------------------------------------------------

tf.reset_default_graph()


# Q and target networks 
q_net = QNetwork(scope="q",VALID_ACTIONS=VALID_ACTIONS)
target_net = QNetwork(scope="target_q", VALID_ACTIONS=VALID_ACTIONS)

# state processor
state_processor = ImageProcess()

# tf saver
saver = tf.train.Saver()


with tf.Session() as sess:
 
      # load model/ initialize model
      if ((train_or_test == 'train' and train_from_scratch == False) or train_or_test == 'test'):
                 latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                 print("loading model ckpt {}...\n".format(latest_checkpoint))
                 saver.restore(sess, latest_checkpoint)
      elif (train_or_test == 'train' and train_from_scratch == True):
                 sess.run(tf.global_variables_initializer())    



      # run
      deep_q_learning(sess, game, q_net=q_net, target_net=target_net, state_processor=state_processor, num_episodes=20000,
                            train_or_test=train_or_test, train_from_scratch=train_from_scratch, start_iter=start_iter, start_episode=start_episode,
                                    replay_memory_size=250000, replay_memory_init_size=5000, update_target_net_every=3000,
                                    gamma=1.0, epsilon_start=epsilon_start, epsilon_end=[0.1,0.001], epsilon_decay_steps=[5e4,5e4], batch_size=64)
        

game.close()
#----------------------------------------------------------------------------------


