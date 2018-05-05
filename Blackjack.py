import gym
import numpy as np
import matplotlib.pyplot as plt
import math

env = gym.make('Blackjack-v0')
env.reset()
learningRate = 3e-6
discount = 0.9999
epochs = 200000
initialValue = 0.4
testEpochs = 1000

def determineAction(observation, Q, bounds, shouldRender):
    #Gets Q at observation for all actions
    curQI = []
    for i in range(len(bounds)):
        #List around the observation[i] because we want to convert output of digitize to scalar value
        curQI.append(np.asscalar(np.digitize(observation[i],bounds[i])))
    curQ = Q[curQI[0]][curQI[1]][curQI[2]][:]

    if shouldRender:
        return np.argmax(curQ)
    else:
        #Uses a weighted random distribution to choose action based on Q values
        cs = np.cumsum(curQ)
        rVal = np.random.uniform(0,cs[-1])
        return np.asscalar(np.digitize(rVal,cs))

def convertObservation(rawObservation):
    if rawObservation[2]:
        usableAce = 1
    else:
        usableAce = 0
    return (rawObservation[0],rawObservation[1],usableAce)

def runEpisode(Q, bounds, shouldRender, env,counts):

    observation = convertObservation(env.reset())

    prevObservation = observation
    for step in range(52):
        action = determineAction(observation, Q, bounds, shouldRender)
        prevObservation = observation
        if shouldRender:
            print(observation,action)
        rawObservation, rawReward, done, info = env.step(action)
        observation = convertObservation(rawObservation)
        reward = (rawReward + 1)/2 # TODO: CHANGE BACK TO: (rawReward + 1)/2 --------------------------------------------------a;skldjfa;lskjdfal;sdf


        updateQ(observation, prevObservation, Q, action, reward, done, bounds,counts)

        if done:
            return reward

def updateQ(observation, prevObservation, Q, action, reward, done, bounds,counts):

    prevQI = []
    for i in range(len(bounds)):
        prevQI.append(np.asscalar(np.digitize(prevObservation[i],bounds[i])))

    #print(prevQI)

    curQI = []
    for i in range(len(bounds)):
        curQI.append(np.asscalar(np.digitize(observation[i],bounds[i])))

    if done:
        Q[prevQI[0]][prevQI[1]][prevQI[2]][action] = (1-learningRate)*Q[prevQI[0]][prevQI[1]][prevQI[2]][action] + learningRate * reward #
    else:
        maxQ = np.max(Q[curQI[0]][curQI[1]][curQI[2]][:])
        #print("maxQ:" + str(maxQ))
        #print("prevQ:" + str(Q[prevQI[0]][prevQI[1]][prevQI[2]][action]))
        #print("curQI:" + str(curQI))
        Q[prevQI[0]][prevQI[1]][prevQI[2]][action] = (1-learningRate)*Q[prevQI[0]][prevQI[1]][prevQI[2]][action] + learningRate * discount * maxQ
    counts[prevQI[0]][prevQI[1]][prevQI[2]] += 1

def main():
    #Has bucket dividers and upper bound but no lowerbound.
    #Q array sizes based on length of each bounds array
    #bounds of [-3,0,3,1e10] will result in Q dimension of 4 and buckets of:
    #-inf to -3, -3 to 0, 0 to 3, and 3 to 1e10
    #Inclusion or exclusion of points on the boundary is unclear

    bounds = []
    bounds.append(np.linspace(2.5,21.5,20))#Your Cards Value
    bounds.append(np.linspace(1.5,11.5,11))#Dealer Card value
    bounds.append(np.array([0.5,1.5]))#Usable Ace


    Q = np.ones((bounds[0].shape[0], bounds[1].shape[0], bounds[2].shape[0],2 ))*initialValue
    counts = np.zeros((bounds[0].shape[0], bounds[1].shape[0], bounds[2].shape[0]))
    totalReward = 0
    #runEpisode(Q,bounds,True,env)
    for i in range(epochs):
        runEpisode(Q,bounds,False,env,counts)
    for i in range(testEpochs):
        totalReward += runEpisode(Q,bounds,True,env,counts)
        #print("counts: ", counts)
        #print("Q: ", Q)
        print("||")
    print("meanReward: ", totalReward/testEpochs)

    plt.figure()
    #print(counts)
    for uInd in range(bounds[2].shape[0]):
        plt.subplot(2,1,uInd+1)
        print("meanCounts: ", np.mean(counts))
        c = np.minimum(counts[:,:,uInd].T,50)
        print("shape of c: ",c.shape)
        print("shape of counts: ", counts.shape)
        plt.imshow(c / np.max(c), cmap='hot', interpolation='nearest')
        plt.title("usableAce:" + str(bounds[2][uInd-1]) + ", "+ str(bounds[2][uInd] ))
        #plt.ylim((bounds[3][0]*180/math.pi,bounds[3][-1]*180/math.pi))
        #plt.xlim((bounds[2][0]*180/math.pi,bounds[2][-1]*180/math.pi))
        plt.xlabel("yourTotal-2")
        plt.ylabel("dealerTotal-1")

    plt.figure()
    for uInd in range(bounds[2].shape[0]):
        action = np.argmax(Q[:,:,uInd,:],axis = 2)
        print("Best Action: ", action)
        plt.subplot(2,1,uInd+1)
        plt.imshow(action.T, cmap='hot', interpolation='nearest')
        plt.title("usableAce:" + str(bounds[2][uInd-1]) + ", "+ str(bounds[2][uInd] ))
        #plt.ylim((bounds[3][0]*180/math.pi,bounds[3][-1]*180/math.pi))
        #plt.xlim((bounds[2][0]*180/math.pi,bounds[2][-1]*180/math.pi))
        plt.xlabel("yourTotal-2")
        plt.ylabel("dealerTotal-1")

    plt.show()
if __name__ == '__main__':
    main()
