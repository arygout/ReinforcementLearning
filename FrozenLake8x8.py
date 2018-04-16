import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

def initState(state):
    state["q"] = np.ones((8,8,4))*0.1
    state["prevObservation"] = ""

def convertObservation(observation):
    curRow = int(observation/8)
    curCol = observation%8
    return (curRow,curCol)


def trainDetermineAction(observation,state):
    row, col = convertObservation(observation)

    cs = np.cumsum(state["q"][row][col][:])
    v = np.random.uniform(0, np.sum( state["q"][row][col][:] ))
    for i in range(cs.shape[0]):
        if v < cs[i]:
            return i

def testDetermineAction(observation,state):
    row, col = convertObservation(observation)
    return np.argmax(state["q"][row][col][:])
    #return np.argmax(state["q"],axis = 2)[row][col]

def updateQ(observation,reward,done,state,a):
    prevRow,prevCol = convertObservation(state["prevObservation"])
    row,col = convertObservation(observation)

    learningRate = 0.03
    discount = 0.98

    oldQ = state["q"][prevRow][prevCol][a]
    maxQ = np.max(state["q"][row][col][:])

    if done:
        state["q"][prevRow][prevCol][a] = oldQ * (1-learningRate) + learningRate * reward
    else:
        state["q"][prevRow][prevCol][a] = oldQ * (1-learningRate) + learningRate * discount*maxQ

def runEpisode(env,state,shouldRender):
    observation = env.reset()
    state["prevObservation"] = observation
    while True:
        a = 0
        if shouldRender:
            a = testDetermineAction(observation,state)
        else:
            a = trainDetermineAction(observation,state)

        state["prevObservation"] = observation
        observation, reward, done, info = env.step(a)
        updateQ(observation,reward,done,state,a)
        if shouldRender:
            env.render()
        if done:
            break




def main():
    state = {}
    initState(state)
    holes = {0:[],
             1:[],
             2:[3],
             3:[5],
             4:[3],
             5:[1,2,6],
             6:[1,4,6],
             7:[3]
             }
    register(
        id='FrozenLakeArya-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '8x8'},
        max_episode_steps=100,
    )
    env = gym.make('FrozenLakeArya-v0')
    for i in range(50000):
        if i%1000 == 0:
            print(i)
        runEpisode(env,state,False)

    for i in range(4):
        runEpisode(env,state,True)
    print(state["q"])
    print()
    bestActions = np.argmax(state["q"],axis = 2)
    for i in range(8):
        line = ""
        for j in range(8):
            if j in holes[i]:
                line += "H"
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
