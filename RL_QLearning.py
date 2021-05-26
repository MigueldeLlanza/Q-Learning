import numpy as np
import random
import pygame
import time
import matplotlib.pyplot as plt

class q_learning():
    def __init__(self, rows=10, cols=10, width=600, height=500, background_colour=(25, 25, 25)):
        self.rows = rows
        self.cols = cols
        self.background_colour = background_colour
        self.width = width
        self.height = height
        self.dimCW = self.width / rows
        self.dimCH = self.height / (cols + 2)
        self.world = np.zeros((rows, cols))
        self.ALLOW_MOV = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1]),
                             4: np.array([0, 0]), 5: np.array([0, 0])}
        self.possible_states = list(np.ndindex(*self.world.shape))
        self.locs = [(rows-1,cols-1),(0,0)]
        self.num_states = rows * cols * len(self.locs)
        self.states = {state: {action: []
                     for action in self.ALLOW_MOV.keys()} for state in range(self.num_states)}

        self.current_state = np.random.choice(np.arange(0, self.num_states, 2))
        self.q_table = np.zeros([self.num_states, len(list(self.ALLOW_MOV.keys()))])
        self.barriers = [(rows // 2, cols // 2), (rows // 2, cols // 2 - 1), (rows // 2, cols // 2 + 1),
                         (rows // 2, cols // + 2)]


    def environment(self):
        ar = []
        for row in range(self.rows):
            for col in range(self.cols):
                for goal in range(len(self.locs)):
                    goal_1 = False
                    done = False
                    for action in self.ALLOW_MOV.keys():
                        reward = -1
                        prev_pos = (row,col)
                        next_pos = tuple(np.asarray([row, col]) + self.ALLOW_MOV[action])
                        state = self.encode(row, col, goal)
                        if self.allowed_action(next_pos):

                            if goal == 1 or action == 4 and next_pos == self.locs[0]:
                                goal_1 = True

                            if action == 4 and next_pos != self.locs[0]:
                                reward = -10

                            if action == 5 and not goal_1:
                                reward = -10

                            if action == 5 and next_pos == self.locs[1] and goal_1 == True:
                                reward = 20
                                done = True

                            if next_pos in self.barriers:
                                reward = -50

                        else:
                            next_pos = (row,col)

                        new_state = self.encode(next_pos[0], next_pos[1], goal_1)

                        self.states[state][action].extend(
                            (1.0, prev_pos, next_pos, new_state, reward, done))
                        if goal_1:
                            ar.append(state)

        return self.states

    def encode(self, row, col, goal):
        return (row * self.rows + col) * 2 + goal

    def allowed_action(self,next_state):
        if next_state in self.possible_states:
            return True
        else:
            return False

    def step(self,state,action):
        prob, prev_pos, next_pos, next_state, reward, done = self.states[state][action]

        return prob, prev_pos, next_pos, next_state, reward, done

    def train_agent(self, iters=1000, alpha=0.1, gamma=0.5, epsilon=0.1):
        """
          Q-Learning
        """

        # For plotting metrics
        episode_rewards = np.zeros(iters + 1)
        xs = list(range(iters + 1))

        for i in range(iters + 1):
            state = np.random.choice(self.num_states)
            epochs, penalties, reward = 0, 0, 0
            done = False
            rewards = []
            while not done:

                if random.uniform(0, 1) < epsilon:
                    action = np.random.choice(list(self.ALLOW_MOV.keys()))  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values

                prob, prev_pos, next_pos, next_state, reward, done = self.step(state, action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                state = next_state

                episode_rewards[i] += reward
                rewards.append(reward)

            if i % 100 == 0:
                print(f"Episode: {i}")

        plt.plot(xs,episode_rewards)
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.title('Rewards over time')
        plt.ylim(-4000,500)
        plt.show()

        return self.q_table


    def polygon(self, x, y):
        self.poly = [((x) * self.dimCW, y * self.dimCH),
                     ((x + 1) * self.dimCW, y * self.dimCH),
                     ((x + 1) * self.dimCW, (y + 1) * self.dimCH),
                     ((x) * self.dimCW, (y + 1) * self.dimCH)]

        return self.poly

    def run_game(self,q_table):

        screen = pygame.display.set_mode((self.width, self.height))
        pygame.init()  # now use display and fonts
        myFont = pygame.font.SysFont("Times New Roman", 18)
        actions = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "Goal 1", 5: "Goal 2"}

        first_goal = False
        done = False
        running = True
        while running:
            new_world = np.copy(self.world)
            screen.fill(self.background_colour)
            time.sleep(0.2)

            if not done:
                action = np.argmax(q_table[self.current_state])
                prob, prev_pos, next_pos, next_state, reward, done = self.states[self.current_state][action]

                new_world[prev_pos[0], prev_pos[1]] = 0
                new_world[next_pos[0], next_pos[1]] = 1

                # Define barriers in the array
                for b in self.barriers:
                    new_world[b] = 4

                if tuple(next_pos) == (self.rows - 1, self.cols - 1) and not first_goal:
                    first_goal = True
                    new_world[self.rows - 1, self.cols - 1] = 3
                    new_world[0, 0] = 2
                elif first_goal:
                    if done:
                        new_world[self.rows - 1, self.cols - 1] = 3
                        new_world[0, 0] = 3

                    else:
                        new_world[self.rows - 1, self.cols - 1] = 3
                        new_world[0, 0] = 2
                else:
                    new_world[self.rows - 1, self.cols - 1] = 2

                self.current_state = next_state

            # pass a string to myFont.render
            action_label = myFont.render("Action: " + actions[action], 1, (255, 255, 255))
            reward_label = myFont.render("Reward: " + str(reward), 1, (255, 255, 255))
            f_goal_label = myFont.render("First Goal: " + str(first_goal), 1, (255, 255, 255))
            s_goal_label = myFont.render("Second Goal: " + str(done), 1, (255, 255, 255))
            completed_label = myFont.render("Task Completed!", 1, (255, 255, 255))

            screen.blit(action_label, (10, 340))
            screen.blit(reward_label, (10, 360))
            screen.blit(f_goal_label, (140, 340))
            screen.blit(s_goal_label, (140, 360))
            if done:
                screen.blit(completed_label, (370, 340))


            for y in range(self.rows):
                for x in range(self.cols):

                    if new_world[x, y] == 1:
                        pygame.draw.polygon(screen, (255, 255, 255), self.polygon(x, y), 0)

                    elif new_world[x, y] == 2:
                        pygame.draw.polygon(screen, (255, 17, 0), self.polygon(x, y), 0)

                    elif new_world[x, y] == 3:
                        pygame.draw.polygon(screen, (100, 255, 100), self.polygon(x, y), 0)

                    elif new_world[x, y] == 4:
                        pygame.draw.polygon(screen, (100, 0, 20), self.polygon(x, y), 0)

            self.world = np.copy(new_world)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False


q_l = q_learning()
q_l.environment()
q_table = q_l.train_agent(1000,alpha=0.1,gamma=0.9,epsilon=0.1)
q_l.run_game(q_table=q_table)