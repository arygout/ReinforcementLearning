import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

def initState(state):
    state["q"] = np.ones((4,4,4))*0.1
    state["prevObservation"] = ""

#Changes the observation from a 0 to 15 number to a tuple of (0 to 4,0 to 4)
def convertObservation(observation):
    curRow = int(observation/4)
    curCol = observation%4
    return (curRow,curCol)

#Creates a weighted random distribution to choose actions based upon their Q values
def trainDetermineAction(observation,state):
    row, col = convertObservation(observation)

    cs = np.cumsum(state["q"][row][col][:])
    v = np.random.uniform(0, np.sum( state["q"][row][col][:] ))
    for i in range(cs.shape[0]):
        if v < cs[i]:
            return i

#Finds the best action at a given state and returns it
def testDetermineAction(observation,state):
    row, col = convertObservation(observation)
    return np.argmax(state["q"][row][col][:]
    #return np.argmax(state["q"],axis = 2)[row][col]

#Updates the Q values by using the current observation, reward, state, and action, and the done boolean
def updateQ(observation,reward,done,state,a):
    prevRow,prevCol = convertObservation(state["prevObservation"])
    row,col = convertObservation(observation)

    #Sets the learning rate and discount
    learningRate = 0.02
    discount = 0.987

    #Gets the Q value of the current state action pair
    oldQ = state["q"][prevRow][prevCol][a]

    #Gets the maximum Q value of the current observation
    maxQ = np.max(state["q"][row][col][:])

    if done:
        #If the program is done it will filter in the reward value
        state["q"][prevRow][prevCol][a] = oldQ * (1-learningRate) + learningRate * reward
    else:
        #If its not done it filters in the maxQ value multiplied by the discount factor
        state["q"][prevRow][prevCol][a] = oldQ * (1-learningRate) + learningRate * discount*maxQ

#Runs an episode
def runEpisode(env,state,shouldRender):
    #Resets environment and gets initial observation
    observation = env.reset()
    #Stores the current state
    state["prevObservation"] = observation
    #Really only goes for a hundred steps because then done gets set to true
    while True:
        a = 0
        #When rendering you want to always take the best action but when training you want to take an action based on a
        #weighted random distribution. This runs either trainDetermineAction or testDetermineAction
        if shouldRender:
            a = testDetermineAction(observation,state)
        else:
            a = trainDetermineAction(observation,state)

        state["prevObservation"] = observation #Stores the previous observation
        observation, reward, done, info = env.step(a) #Takes a step with the action that was just determined
        updateQ(observation,reward,done,state,a) #Updates the Q values
        #Determines if it should render or not render
        if shouldRender:
            env.render()
        #Stops the episode if the done condition is met
        if done:
            break




def main():
    #Creates and initializes the state
    state = {}
    initState(state)
    #Sets a couple of environment parameters
    register(
        id='FrozenLakeArya-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4'},
        max_episode_steps=100,
    )
    #Makes the environment and runs 10000 episodes
    env = gym.make('FrozenLakeArya-v0')
    for i in range(10000):
        print(i)
        runEpisode(env,state,False)

    #Defines an array with all the holes
    holes = {
    0:[],
    1:[1,3],
    2:[3],
    3:[0]
    }
    #Renders an episode with the best results
    runEpisode(env,state,True)
    #print(state["q"])
    print()
    #Finds the best action to take at every state and creates a chart showing this. It also shows the location of all the holes
    bestActions = np.argmax(state["q"],axis = 2)
    for i in range(4):
        line = ""
        for j in range(4):
            if j in holes[i]:
                line +="H"
            elif bestActions[i][j] == 0:
                line += "<"
            elif bestActions[i][j] == 1:
                line += "v"
            elif bestActions[i][j] == 2:
                line += ">"
            elif bestActions[i][j] == 3:
                line += "^"
        print(line)


if __name__ == '__main__':
    main()
