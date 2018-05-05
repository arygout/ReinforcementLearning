import gym
import numpy as np
import matplotlib.pyplot as plt
import math

env = gym.make('CartPole-v1')
env.reset()
learningRate = 0.01
discount = 0.9999
epochs = 5000

def determineAction(observation, Q, bounds, shouldRender):
    #Gets Q at observation for all actions
    curQI = []
    for i in range(len(bounds)):
        #List around the observation[i] because we want to convert output of digitize to scalar value
        curQI.append(np.asscalar(np.digitize(observation[i],bounds[i])))
    curQ = Q[curQI[0]][curQI[1]][curQI[2]][curQI[3]][:]

    if shouldRender:
        return np.argmax(curQ)
    else:
        #Uses a weighted random distribution to choose action based on Q values
        cs = np.cumsum(curQ)
        rVal = np.random.uniform(0,cs[-1])
        return np.asscalar(np.digitize(rVal,cs))

def runEpisode(Q, bounds, shouldRender, env,counts):

    observation = env.reset()
    prevObservation = observation
    total_reward = 0
    for step in range(200):
        action = determineAction(observation, Q, bounds, shouldRender)
        prevObservation = observation
        observation, reward, done, info = env.step(action)
        #print(observation)
        total_reward += reward

        updateQ(observation, prevObservation, Q, action, total_reward, done, bounds,counts)
        if shouldRender:
            env.render()
        if done:
            break
    if shouldRender:
        print("step: " + str(step))
    return total_reward

def updateQ(observation, prevObservation, Q, action, reward, done, bounds,counts):

    prevQI = []
    for i in range(len(bounds)):
        prevQI.append(np.asscalar(np.digitize(prevObservation[i],bounds[i])))

    #print(prevQI)

    curQI = []
    for i in range(len(bounds)):
        curQI.append(np.asscalar(np.digitize(observation[i],bounds[i])))

    if done:
        Q[prevQI[0]][prevQI[1]][prevQI[2]][prevQI[3]][action] = (1-learningRate)*Q[prevQI[0]][prevQI[1]][prevQI[2]][prevQI[3]][action] + learningRate * reward #
    else:
        maxQ = np.max(Q[curQI[0]][curQI[1]][curQI[2]][curQI[3]][:])
        #print("maxQ:" + str(maxQ))
        #print("prevQ:" + str(Q[prevQI[0]][prevQI[1]][prevQI[2]][prevQI[3]][action]))
        #print("curQI:" + str(curQI))
        Q[prevQI[0]][prevQI[1]][prevQI[2]][prevQI[3]][action] = (1-learningRate)*Q[prevQI[0]][prevQI[1]][prevQI[2]][prevQI[3]][action] + learningRate * discount * maxQ
    counts[prevQI[0]][prevQI[1]][prevQI[2]][prevQI[3]] += 1

def main():
    #Has bucket dividers and upper bound but no lowerbound.
    #Q array sizes based on length of each bounds array
    #bounds of [-3,0,3,1e10] will result in Q dimension of 4 and buckets of:
    #-inf to -3, -3 to 0, 0 to 3, and 3 to 1e10
    #Inclusion or exclusion of points on the boundary is unclear

    bounds = []
    bounds.append(np.array([1e3]))#Position
    bounds.append(np.array([-0.5, -0.2, 0.2,  0.5,1e10]))#Velocity
    #bounds.append(np.array([0,1e3]))#Velocity
    bounds.append(np.array([-8, -4, 0, 4, 8, 1e10])*math.pi/180)#Angle
    #bounds.append(np.array([0,1e3])*math.pi/180)#Angle
    bounds.append(np.array([-0.6, -0.3, -0.15, 0.15, 0.3, 0.6,1e3]))#Angular Velocity
    #bounds.append(np.array([-0.3,0,0.3,1e10]))#Angular Velocity

    Q = np.ones((bounds[0].shape[0], bounds[1].shape[0], bounds[2].shape[0], bounds[3].shape[0],2 ))*40
    counts = np.zeros((bounds[0].shape[0], bounds[1].shape[0], bounds[2].shape[0], bounds[3].shape[0]))

    #runEpisode(Q,bounds,True,env)
    for i in range(epochs):
        runEpisode(Q,bounds,False,env,counts)
    for i in range(5):
        runEpisode(Q,bounds,True,env,counts)

    """for vInd in range(bounds[1].shape[0]):
        print("velocity:" + str(bounds[1][vInd]))
        print(" 1234567890")
        for wInd in range(bounds[3].shape[0]):
            line = str(wInd) + ""
            for tInd in range(bounds[2].shape[0]):
                action = np.argmax(Q[0][vInd][tInd][wInd][:])
                if Q[0][vInd][tInd][wInd][0] == 40:
                    line += " "
                elif Q[0][vInd][tInd][wInd][1] == 40:
                    line += " "
                elif action == 0:
                    line += "L"
                elif action == 1:
                    line += "."
                else:
                    line+= "ERROR"
            print(line)"""

    plt.figure()
    for vInd in range(bounds[1].shape[0]):
        action = np.argmax(Q[0][vInd][:][:][:],axis = 2)
        plt.subplot(2,3,vInd+1)
        plt.imshow(action.T, cmap='hot', interpolation='nearest')
        plt.title("velocity:" + str(bounds[1][vInd-1]) + ", "+ str(bounds[1][vInd] ))
        #plt.ylim((bounds[3][0]*180/math.pi,bounds[3][-1]*180/math.pi))
        #plt.xlim((bounds[2][0]*180/math.pi,bounds[2][-1]*180/math.pi))
        plt.xlabel("angle")
        plt.ylabel("angular velocity")
    plt.figure()
    #print(counts)
    for vInd in range(bounds[1].shape[0]):
        plt.subplot(2,3,vInd+1)
        c = np.minimum(counts[0][vInd][:][:].T,50)
        plt.imshow(c / np.max(c), cmap='hot', interpolation='nearest')
        plt.title("velocity:" + str(bounds[1][vInd-1]) + ", "+ str(bounds[1][vInd] ))
        #plt.ylim((bounds[3][0]*180/math.pi,bounds[3][-1]*180/math.pi))
        #plt.xlim((bounds[2][0]*180/math.pi,bounds[2][-1]*180/math.pi))
        plt.xlabel("angle")
        plt.ylabel("angular velocity")
    plt.show()
if __name__ == '__main__':
    main()
