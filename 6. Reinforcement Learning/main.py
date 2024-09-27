from snake import *
from utility import *
from cube import *

import pygame
import numpy as np
from tkinter import messagebox
from snake import Snake
import matplotlib.pyplot as plt

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    snake_1 = Snake((255, 0, 0), (15, 15), SNAKE_1_Q_TABLE)
    snake_2 = Snake((255, 255, 0), (5, 5), SNAKE_2_Q_TABLE)
    snake_1.addCube()
    snake_2.addCube()

    snack = Cube(randomSnack(ROWS, snake_1), color=(0, 255, 0))

    clock = pygame.time.Clock()
    episode_total_reward_1 = 0
    episode_total_reward_2 = 0
    
    episode_rewards_1 = []
    episode_rewards_2 = []
    num_win1 = 0
    num_win2 = 0
    total_episodes = 0
    while True:
        reward_1 = 0
        reward_2 = 0
        pygame.time.delay(15)
        clock.tick(60)
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                if messagebox.askyesno("Quit", "Do you want to save the Q-tables?"):
                    save(snake_1, snake_2)
                pygame.quit()
                print('Snake 1 wins:', num_win1)
                print('Snake 2 wins:', num_win2)
                print('draw games:', total_episodes - num_win1 - num_win2)
                print('total episodes:', total_episodes)
                plt.figure()
                plt.plot(episode_rewards_1, label='Snake 1 Reward per Episode')
                plt.plot(episode_rewards_2, label='Snake 2 Reward per Episode')
                plt.legend(loc='upper left')
                plt.title('Total Rewards Per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.show()
                return
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                np.save(SNAKE_1_Q_TABLE, snake_1.q_table)
                np.save(SNAKE_2_Q_TABLE, snake_2.q_table)
                pygame.time.delay(1000)

        state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
        state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)
        
        first_win1, first_win2 = False, False
        snack, reward_1, finished1, win_1, win_2 = snake_1.calc_reward(snack, snake_2)
        first_win1 = win_1
        first_win2 = win_2
        snack, reward_2, finished2, win_2, win_1 = snake_2.calc_reward(snack, snake_1)
        if first_win1 or win_1:
            num_win1 += 1
        if first_win2 or win_2:
            num_win2 += 1

        snake_1.update_q_table(state_1, action_1, new_state_1, reward_1)
        snake_2.update_q_table(state_2, action_2, new_state_2, reward_2)
  
        episode_total_reward_1 += reward_1
        episode_total_reward_2 += reward_2
        redrawWindow(snake_1, snake_2, snack, win)
  
        if finished1 or finished2:
            total_episodes += 1

            episode_rewards_1.append(episode_total_reward_1)
            episode_rewards_2.append(episode_total_reward_2)
            episode_total_reward_1 = 0
            episode_total_reward_2 = 0


if __name__ == "__main__":
    main()