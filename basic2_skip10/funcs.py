import numpy as np
import sys
import tensorflow as tf

#from vizdoom import DoomGame, ScreenResolution
from vizdoom import *


# convert raw RGB image of size 480x640x3/240x320x3 into 84x84 grayscale image
class ImageProcess():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            #self.input_state = tf.placeholder(shape=[480, 640, 3], dtype=tf.uint8)
            self.input_state = tf.placeholder(shape=[240, 320, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 90, 0, 240-90, 320)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })



# copy params from qnet1 to qnet2
def copy_model_parameters(sess, qnet1, qnet2):
    q1_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet1.scope)]
    q1_params = sorted(q1_params, key=lambda v: v.name)
    q2_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet2.scope)]
    q2_params = sorted(q2_params, key=lambda v: v.name)
    update_ops = []
    for q1_v, q2_v in zip(q1_params, q2_params):
        op = q2_v.assign(q1_v)
        update_ops.append(op)
    sess.run(update_ops)


# epsilon-greedy
def epsilon_greedy_policy(qnet, num_actions):
    def policy_fn(sess, observation, epsilon):
        if (np.random.rand() < epsilon):  
          # explore: equal probabiities for all actions
          A = np.ones(num_actions, dtype=float) / float(num_actions)
        else:
          # exploit 
          q_values = qnet.predict(sess, np.expand_dims(observation, 0))[0]
          max_Q_action = np.argmax(q_values)
          A = np.zeros(num_actions, dtype=float)
          A[max_Q_action] = 1.0 
        return A
    return policy_fn



# populate replay memory
def populate_replay_mem(sess, game, state_processor, replay_memory_init_size, policy, epsilon_start, epsilon_end, epsilon_decay_steps, VALID_ACTIONS, Transition):

    game.new_episode()
    game_state = game.get_state()
    game_var = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        
    state_img = game_state.screen_buffer # new_state  
    state_img = np.rollaxis(state_img,0,3)   

    state = state_processor.process(sess, state_img)
    state = np.stack([state] * 4, axis=2)

    delta_epsilon = (epsilon_start - epsilon_end)/float(epsilon_decay_steps)

    replay_memory = []

    for i in range(replay_memory_init_size):
        epsilon = max(epsilon_start - float(i) * delta_epsilon, epsilon_end)
        action_probs = policy(sess, state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

#----------------------

        game.set_action(action_list(action))
        frame_per_action = 10   
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

#-------------

        if (not done):
          # non-Terminal state
          next_state_img = new_game_state.screen_buffer # new_state  
          next_state_img = np.rollaxis(next_state_img,0,3) 
          next_state_img = state_processor.process(sess, next_state_img)  
          
          next_state = np.zeros((84,84,4),dtype=np.uint8)
          next_state[:,:,0] = state[:,:,1] 
          next_state[:,:,1] = state[:,:,2]
          next_state[:,:,2] = state[:,:,3]
          next_state[:,:,3] = next_state_img         
        else: 
          # Terminal state
          next_state = state
             

        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done:
            game.new_episode()
            game_state = game.get_state()
            game_var = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
        
            state_img = game_state.screen_buffer # new_state  
            state_img = np.rollaxis(state_img,0,3)   

            state = state_processor.process(sess, state_img)
            state = np.stack([state] * 4, axis=2) 
        else:
            state = next_state
            game_var = new_game_var 
    return replay_memory



def doom_game():

  game = DoomGame()
  game.load_config("../scenarios/basic.cfg") 
  #game.load_config("../scenarios/defend_the_center.cfg")
  #game.set_doom_map("map01")
  game.set_screen_resolution(ScreenResolution.RES_320X240)
  #game.set_screen_resolution(ScreenResolution.RES_640X480)
  game.set_render_hud(False)
  game.set_render_crosshair(False)
  game.set_render_weapon(True)
  game.set_render_decals(False)
  game.set_render_particles(False)
  game.add_available_button(Button.MOVE_LEFT) # basic
  game.add_available_button(Button.MOVE_RIGHT) # basic
  #game.add_available_button(Button.TURN_LEFT) # dtc
  #game.add_available_button(Button.TURN_RIGHT) # dtc
  game.add_available_button(Button.ATTACK)
  game.set_episode_timeout(300) # 2100) # basic: 300; dtc: 2100
  game.set_episode_start_time(10)
  game.set_window_visible(True) 
  game.set_sound_enabled(False)
  game.set_living_reward(-1) # 0.2) # 0.2 for dtc; -1 for basic
  game.set_mode(Mode.PLAYER)
  game.init()
  return game



def shape_reward(r_t, new_game_var, game_var):
      return r_t # for basic scenario

      # bonus for kill
      if (new_game_var[0] > game_var[0]): 
          #print("kill ", r_t)
          #print(game_var, new_game_var)  
          r_t = r_t + 2.0 

      # penalty for using up ammo
      if (new_game_var[1] < game_var[1]):
          #print("ammo ", r_t)
          #print(game_var, new_game_var) 
          r_t = r_t - 0.1 

      # penalty for loss of health 
      if (new_game_var[2] < game_var[2]):
          #print("health ", r_t)
          #print(game_var, new_game_var)
          r_t = r_t - 0.1
      return r_t



def action_list(a):
   if a == 0:
      return [1, 0, 0]
   elif a == 1:
      return [0, 1, 0]
   elif a == 2:
      return [0, 0, 1]
   else:
     print("action_list error ")
     sys.exit()

