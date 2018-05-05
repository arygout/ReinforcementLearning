import gym
import numpy as np
import matplotlib.pyplot as plt

def runEpisode(env):
    observation = env.reset()
    for i in range(52):
        action = int(np.round(np.random.uniform(0,1)))
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            return (reward + 1)/2
    print("SOMETHING WRONG IN CODE")
    return -1

def main():
  env = gym.make('Blackjack-v0')
  epochs = 10000
  rewards = np.zeros((epochs,1))

  for i in range(epochs):
      rewards[i] = runEpisode(env)
      print(" ")

  print("rewards: ",rewards)
  print(np.mean(rewards))


if __name__ == '__main__':
    main()
