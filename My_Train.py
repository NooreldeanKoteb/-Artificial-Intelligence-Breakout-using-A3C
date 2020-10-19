# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:43:21 2020

@author: Nooreldean Koteb
"""

#Training the AI
import torch
import torch.nn.functional as F
from envs import create_atari_env
from My_Model import ActorCritic
from torch.autograd import Variable

#May not be important, but here to make sure it works properly
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


#Train function
#(shift each seed for each agent ,enviroment paramenters, model, optimizer)
def train(rank, params, shared_model, optimizer):
    #desyncronize each training agent  
    torch.manual_seed(params.seed + rank)
    
    #Get the enviroment
    env = create_atari_env(params.env_name)
    
    #Align each seed of the enviroment to one of the agents
    env.seed(params.seed + rank)
    
    #Get our model (A3C brain)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    #Perpare input states (input black and white images 42x42)
    #Get numpy array - Initalize state
    state = env.reset()
    #Convert to torch.tensors
    state = torch.from_numpy(state)
    #Signals if the game is over
    done = True
    
    #length of one episode
    episode_length = 0
    
    #Loop
    while True:
        #Increment
        episode_length += 1
        
        #Get the shared model
        model.load_state_dict(shared_model.state_dict())
        
        #If game is done reinitialize the hidden and cell nodes
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        
        #Else keep old  data
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
    
    
        #Initialize values, log_probs, rewards, and entropies
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        #Steps through exploration
        for step in range(params.num_steps):
            #Go through Neural Network model    
            #value of critic, action_values of the actor, hidden & cell nodes tuple 
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
        
            #Probabilities
            prob = F.softmax(action_values)
            
            #Log of probabilities
            log_prob = F.log_softmax(action_values)
        
            #Calculate entropy
            entropy = -(log_prob * prob).sum(1)
            
            #Store entropy in entropies list
            entropies.append(entropy)
            
            #Picking the action from probabilities
            action = prob.multinomial().data
            
            #Updating the log probability
            log_prob = log_prob.gather(1, Variable(action))
            
            #Append to values, and log_probs lists
            values.append(value)
            log_probs.append(log_prob)
            
            #play action and get new state, reward, and if game is done
            state, reward, done, _ = env.step(action.numpy())
            
            #Make sure agent isn't stuck in a state
            #If game is done or game is longer then the set max episode length
            done = (done or episode_length >= params.max_episode_length)
            
            #Makes reward between -1 and 1
            reward = max(min(reward, 1), -1)
            
            #If game is done
            if done:
                #Restart game
                episode_length = 0
                state = env.reset()
            
            #Update the state
            state = torch.from_numpy(state)
            
            #Append reward to rewards list
            rewards.append(reward)
            
            #If game is done stop exploration
            if done:
                break
    
        #Initialize cumulative reward
        R = torch.zeros(1, 1)
        
        #If game isnt done
        if not done:         
            #Get just the value from the model
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            
            #Set cumulative reward to the critic value
            R = value.data
        
        #Append value to values list
        values.append(Variable(R))
        
        
        #Calculating loss
        #Initializing loss to zero
        policy_loss = 0
        value_loss = 0
        
        #Making R a Variable
        R =  Variable(R)
        
        #Generalized advantage estimation initialization
        gae = torch.zeros(1, 1) #A(a,s) = Q(a,s) - V(s)
    
        #Moving backwards in the rewards from exploration  |Same as doom
        for i in reversed(range(len(rewards))):
            #Updating the cumulative reward
            #R = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_{n-1} + gamma^nb_steps * V(last_State)
            R = params.gamma * R + rewards[i]
            
            #Compute advantage
            advantage = R - values[i]
            
            #Computing/updating the value loss
            value_loss = value_loss + 0.5 * advantage.pow(2) # Q*(a*,s) = V*(s)
            
            
            #Temporal difference
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            
            #Computing/updating generalized advantage estimation
            #gae = sum_i (gamma*tau)^i * TD(i)
            gae = gae * params.gamma * params.tau + TD  
            
            #Computing/updating the policy loss
            #policy_loss = - sum_i log(pi_i)*gae +0.01*H_i
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]        
        
        
        #Initializing the optimizer
        optimizer.zero_grad() 
        
        #backwards propagation (this gives 0.5(half) as much importance to the policy loss then the value loss)
        (policy_loss + 0.5 * value_loss).backward()
        
        #Make sure gradient wont take extremly large values, the 40 means the gradient stays between 0 and 40
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
		
        #Makes sure the model and the shared model share the same gradient
        ensure_shared_grads(model, shared_model)
        
        #Optimization step to reduce the losses
        optimizer.step()