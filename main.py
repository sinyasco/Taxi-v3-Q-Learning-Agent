import numpy as np
import gymnasium
import random

environment = gymnasium.make('Taxi-v3')

learningRate = 0.9 #Q(s,a) <- Q(s,a) + learningRate*A (learningRate = step)
discountFactor = 0.9 #gamma
epsilonRandom = 1 #start with pure randomness
epsilonDecay = 0.995 #used to decrease epsilonRandom
minEpsilon = 0.01 #at least 0.01 randomness
nbEpisodes = 10000
maxSteps = 100

Q_table = np.zeros((environment.observation_space.n, environment.action_space.n))

def choose_action(state):
    if random.uniform(0,1) < epsilonRandom:
        return environment.action_space.sample()
    else:
        return np.argmax(Q_table[state, :])

for episode in range(nbEpisodes):
    state,_ = environment.reset()
    done = False

    for step in range(maxSteps):
        action = choose_action(state)
        next_state,reward, done, trancuated,_ = environment.step(action)

        old_value = Q_table[state,action]
        nextMax = np.max(Q_table[next_state, :])

        Q_table[state,action] = (1-learningRate)*old_value + learningRate*(reward + discountFactor*nextMax)
        state = next_state

        if done or trancuated:
            break
    epsilonRandom = max(minEpsilon, epsilonRandom*epsilonDecay)

environment = gymnasium.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state,_ = environment.reset()
    done=False

    print('Episode', episode)

    for step in range(maxSteps):
        environment.render()
        action = np.argmax(Q_table[state,:])
        next_state,reward,done,trancuated,_ = environment.step(action)
        state=next_state 

        if done or trancuated:
            environment.render()
            print("Finished Episode : ",episode," with reward : ",reward)
            break
environment.close()




