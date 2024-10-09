import numpy as np
import pygame
import math
import time
import gymnasium as gym
import socket
from gymnasium import spaces
from random import randrange


class TFGym(gym.Env):
    def __init__(self, obs_space, action_space):
        self.timer = None

        # Mouse and Trackpad space is around -140 to 140 - Mouse set to 3200 CPI
        self.observation_space = obs_space #spaces.Box(low=np.array([0, 0]), high=np.array([140, 140]), dtype=np.float32) # Observations: # of counts 
        
        self.action_space = action_space #spaces.Box(low=np.array([0, 0]), high=np.array([5, 5]), dtype=np.float32) # Action Space = speed multiplier

        self.window = None
        self.clock = None # We can figure it out later 
        self.fps = 60

        # Default parameters for ISO Fitts 
        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.cursor_size = 7
        self.max_target_size = 50

        self.cursor = None 
        self.target = None 
        self.current_target_size = randrange(self.cursor_size, self.max_target_size)
        self.start_time = time.time()
        self.og_distance = 0

        self._dx = 0
        self._dy = 0

        self.w = 1000
        self.h = 1000

        self.time_since_render = 0
        self.rewards = [0]
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1/self.fps) 
        self.sock.bind(('127.0.0.1', 12345))

    def step(self, action):
        direction = action * np.array([self._dx, self._dy])

        # Make sure the agent doesn't leave the screen 
        if self._agent_location[0] + direction[0] > 0 + self.cursor_size//2 and self._agent_location[0] + direction[0]  + self.cursor_size//2 < self.w:
            self._agent_location[0] += direction[0]  
        if self._agent_location[1] + direction[1] > 0 + self.cursor_size//2 and self._agent_location[1] + direction[1] + self.cursor_size//2 < self.h:
            self._agent_location[1] += direction[1]
            
        terminated = False
        if self._in_circle():
            if self.timer is None:
                self.timer = time.time()
            elif time.time() - self.timer >= 1:
                throughput = math.log2(self.og_distance/self.current_target_size + 1)/(time.time() - self.start_time - 1)
                terminated = True
                self.timer = None
                self.current_target_size = randrange(self.cursor_size, self.max_target_size)
                self.start_time = time.time()
                self.og_distance = math.dist(self._agent_location, self._target_location)
        else:
            self.timer = None


        if terminated:
            reward = throughput # It is a function of throughput  
        else:
            reward = 0
        # elif not self._in_circle():
        #     reward = -0.01
        # else: 
        #     reward = 0 
        
        self.rewards[-1] += reward
        if terminated:
            self.rewards.append(0)
            np.save('rewards.npy', self.rewards)

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, reward, terminated, False, info

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Randomly place the circle 
        self._agent_location = np.array([randrange(0, self.w-self.cursor_size*2), randrange(0, self.h-self.cursor_size*2)])
        self._target_location = np.array([randrange(0, self.w-self.current_target_size*2), randrange(0, self.h-self.current_target_size*2)])

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def _in_circle(self):
        if self.target:
            return math.sqrt((self.target.centerx - self.cursor.centerx)**2 + (self.target.centery - self.cursor.centery)**2) < (self.target[2]/2 + self.cursor[2]/2)
        else:
            return (self._target_location[0]-self._agent_location[0])**2 + (self._target_location[1]-self._agent_location[1])^2 < (self.current_target_size/2 + self.cursor_size/2)

    def _get_obs(self):
        return np.abs(np.array([self._dx, self._dy]))

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) 
            self.w, self.h = pygame.display.get_surface().get_size()
            self.font = pygame.font.SysFont('helvetica', 40)
            pygame.mouse.set_visible(False)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.w, self.h))
        canvas.fill((255, 255, 255))

        # Draw Circle 
        self.cursor = pygame.draw.circle(canvas, self.RED, (self._target_location[0] + self.current_target_size, self._target_location[1] + self.current_target_size), self.current_target_size)
        # Draw Cursor 
        self.target = pygame.draw.circle(canvas, self.BLACK, (self._agent_location[0] + self.cursor_size, self._agent_location[1] + self.cursor_size), self.cursor_size)

        # Draw Timer 
        if self.timer is not None:
            duration = round((time.time()-self.timer),2)
            time_str = str(duration)
            draw_text = self.font.render(time_str, 1, self.BLACK)
            canvas.blit(draw_text, (10, 10))

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        
        if time.time() - self.time_since_render >= 1/self.fps:
            pygame.display.update()
            self.time_since_render = time.time()

        # We want to wait for a new event from the EMG controller
        try:
            data, _ = self.sock.recvfrom(1024)
            data = str(data.decode("utf-8"))
            self._dx = float(data.split(' ')[0])
            self._dy = float(data.split(' ')[1])
        except:
            self._dx = 0 
            self._dy = 0

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()