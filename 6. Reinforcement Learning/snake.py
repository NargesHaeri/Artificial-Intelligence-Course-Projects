from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np


class Snake:
    body = []
    turns = {}
    action_counter = 0
    old_distance_from_snack = 0
    new_distance_from_snack = 0

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4))

        self.lr = 0.001 # TODO: Learning rate
        self.discount_factor = 0.95 # TODO: Discount factor
        self.epsilon = 0.8 # TODO: Epsilon
        self.min_epsilon = 0.1 # TODO: Minimum epsilon
        self.epsilon_decay = 0.995
        
    def get_optimal_policy(self, state):
        actions = self.q_table[tuple(state)]
        winner = np.argwhere(actions == np.amax(actions)).flatten().tolist()
        rand = random.randint(0, len(winner) - 1)
        return winner[rand]

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)

        self.action_counter += 1
        if self.action_counter == 100:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.action_counter = 0
        return action

    def update_q_table(self, state, action, next_state, reward):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

    def calculate_location_score(self, location, snack, other_snake):
        score = 1  # Default to 1 (empty)
        
        if location in [segment.pos for segment in self.body]:
            score = 0  # Location is part of self body
        elif location in [segment.pos for segment in other_snake.body]:
            score = 0  # Location is part of other snake's body
        elif location == snack.pos:
            score = 1  # Location is the snack
        elif location[0] >= ROWS - 1 or location[0] < 1 or location[1] >= ROWS - 1 or location[1] < 1:
            score = 0  # Location is out of board bounds
        
        return score

    def get_adjacent_locations(self):
        locations = [(0, 0)] * 8
        head_pos = self.head.pos
        
        if self.dirnx == 1:  # Moving right
            locations = [
                (head_pos[0], head_pos[1] + 1),
                (head_pos[0], head_pos[1] - 2),
                (head_pos[0], head_pos[1] - 1),
                (head_pos[0] + 1, head_pos[1] - 1),
                (head_pos[0] + 2, head_pos[1]),
                (head_pos[0] + 1, head_pos[1]),
                (head_pos[0] + 1, head_pos[1] + 1),
                (head_pos[0], head_pos[1] + 2)
            ]
        elif self.dirny == -1:  # Moving up
            locations = [
                (head_pos[0] + 1, head_pos[1]),
                (head_pos[0] - 2, head_pos[1]),
                (head_pos[0] - 1, head_pos[1]),
                (head_pos[0] - 1, head_pos[1] - 1),
                (head_pos[0], head_pos[1] - 2),
                (head_pos[0], head_pos[1] - 1),
                (head_pos[0] + 1, head_pos[1] - 1),
                (head_pos[0] + 2, head_pos[1])
            ]
        elif self.dirnx == -1:  # Moving left
            locations = [
                (head_pos[0], head_pos[1] - 1),
                (head_pos[0], head_pos[1] + 2),
                (head_pos[0], head_pos[1] + 1),
                (head_pos[0] - 1, head_pos[1] + 1),
                (head_pos[0] - 2, head_pos[1]),
                (head_pos[0] - 1, head_pos[1]),
                (head_pos[0] - 1, head_pos[1] - 1),
                (head_pos[0], head_pos[1] - 2)
            ]
        elif self.dirny == 1:  # Moving down
            locations = [
                (head_pos[0] - 1, head_pos[1] - 1),
                (head_pos[0] + 2, head_pos[1]),
                (head_pos[0] + 1, head_pos[1]),
                (head_pos[0] + 1, head_pos[1] + 1),
                (head_pos[0], head_pos[1] + 2),
                (head_pos[0], head_pos[1] + 1),
                (head_pos[0] - 1, head_pos[1] + 1),
                (head_pos[0] - 2, head_pos[1])
            ]
        
        return locations

    def get_current_direction(self):
        if self.dirnx == 1:  # Right
            return 1
        elif self.dirny == -1:  # Up
            return 2
        elif self.dirnx == -1:  # Left
            return 0
        elif self.dirny == 1:  # Down
            return 3

    def get_relative_position(self, point):
        if point.pos[0] <= self.head.pos[0] and point.pos[1] <= self.head.pos[1]:
            return 0  # Northwest
        elif point.pos[0] >= self.head.pos[0] and point.pos[1] <= self.head.pos[1]:
            return 1  # Northeast
        elif point.pos[0] >= self.head.pos[0] and point.pos[1] >= self.head.pos[1]:
            return 2  # Southeast
        elif point.pos[0] <= self.head.pos[0] and point.pos[1] >= self.head.pos[1]:
            return 3  # Southwest

    def get_state(self, snack, other_snake):
        state = [0] * 10
        adjacent_locations = self.get_adjacent_locations()
        
        for i, location in enumerate(adjacent_locations):
            state[i] = self.calculate_location_score(location, snack, other_snake)
        
        state[8] = self.get_relative_position(snack)
        state[9] = self.get_current_direction()
        
        return tuple(state)
    
    def move(self, snack, other_snake):
        state = self.get_state(snack, other_snake) # TODO: Create state
        action = self.make_action(state)

        self.old_distance_from_snack = abs(self.head.pos[0] - snack.pos[0]) + abs(self.head.pos[1] - snack.pos[1])


        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)
        
        new_state = self.get_state(snack, other_snake)

        self.new_distance_from_snack = abs(self.head.pos[0] - snack.pos[0]) + abs(self.head.pos[1] - snack.pos[1]) 

        return state, new_state, action

    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        finished, win_self, win_other = False, False, False
        if self.new_distance_from_snack < self.old_distance_from_snack:
            reward += 2

        if self.check_out_of_board():
            # TODO: Punish the snake for getting out of the board
            reward += -30
            win_other = True
            finished = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            # TODO: Reward the snake for eating
            reward += 25
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # TODO: Punish the snake for hitting itself
            reward += -25
            win_other = True
            reset(self, other_snake)
            finished = True
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                # TODO: Punish the snake for hitting the other snake
                reward += -28
                win_other = True
                finished = True

            else:
                if len(self.body) > len(other_snake.body):
                    # TODO: Reward the snake for hitting the head of the other snake and being longer
                    reward += 10
                    win_self = True
                    finished = True

                elif len(self.body) == len(other_snake.body):
                    finished = True
                    # TODO: No winner
                    

                else:
                    # TODO: Punish the snake for hitting the head of the other snake and being shorter
                    reward += -20
                    win_other = True
                    finished = True
                    
            reset(self, other_snake)
            
        return snack, reward, finished, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        