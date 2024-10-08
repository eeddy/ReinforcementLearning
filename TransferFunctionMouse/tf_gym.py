import numpy as np
import pygame
import math
import time
import gymnasium as gym
import socket
from gymnasium import spaces
from random import randrange


class TFGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.window_size = 1000
        self.timer = None

        # PPI = 109 = 430 pixels (per meter)
        self.PPI = 430

        # Mouse and Trackpad space is around -140 to 140 - Mouse set to 3200 CPI
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([5, 5]), dtype=np.float32) # Gain (speed multiplier)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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

        self._last_dist = 1000000 
        self._dx = 0
        self._dy = 0

        self.time_since_render = 0
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1/self.fps) 
        self.sock.bind(('127.0.0.1', 12345))

    def step(self, action):
        velocity = action
        direction = np.array([self._dx, self._dy]) * velocity * self.PPI

        # Make sure the agent doesn't leave the screen 
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size - 10
        )

        terminated = False
        if self._in_circle():
            if self.timer is None:
                self.timer = time.time()
            elif time.time() - self.timer >= 1:
                terminated = True
                self.timer = None
                self.current_target_size = randrange(self.cursor_size, self.max_target_size)
        else:
            self.timer = None

        new_dist = math.dist(self._agent_location, self._target_location)

        if terminated:
            reward = 1
        elif self._last_dist <= math.dist(self._agent_location, self._target_location) and not self._in_circle():
            reward = -0.1
        else:
            reward = 0.1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._last_dist = new_dist

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly place the circle 
        self._agent_location = self.np_random.integers(0, self.window_size-self.cursor_size*2, size=2, dtype=int)
        self._target_location = self.np_random.integers(0, self.window_size-self.current_target_size*2, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
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

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) #pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont('helvetica', 40)
            pygame.mouse.set_visible(False)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
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

        if self.render_mode == "human":
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