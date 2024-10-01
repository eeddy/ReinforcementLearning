import numpy as np
import pygame
import math
import time
import gymnasium as gym
import socket
from gymnasium import spaces


class TFGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.window_size = 750
        self.timer = None

        self.observation_space = spaces.Dict(
            {
                "class": spaces.Discrete(5),
                "probs": spaces.Box(0, 1, shape=(5,), dtype=np.float32),
                "mav": spaces.Box(0, 1, shape=(8,), dtype=np.float32),
            }
        ) 
        
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1.0]), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None # We can figure it out later 
        SPEED = 30
        self._action_to_direction = {
            1: np.array([0, -1*SPEED]),
            0: np.array([0, SPEED]),
            2: np.array([0,0]),
            4: np.array([-1*SPEED, 0]),
            3: np.array([SPEED, 0]),
        }

        # Default parameters for ISO Fitts 
        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.cursor_size = 7
        self.target_size = 50

        self.cursor = None 
        self.target = None 

        self._last_dist = 1000000 
        self._last_mav = np.zeros(8)
        self._last_class = 0
        self._last_probs = np.zeros(5)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 12346))

    def step(self, action):
        velocity = action 
        direction = self._action_to_direction[self._last_class] * velocity

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
        else:
            self.timer = None

        if terminated:
            reward = 1 
        elif self._in_circle():
            reward = 0.1
        elif self._last_dist <= math.dist(self._agent_location, self._target_location):
            reward = -0.1
        else:
            reward = 0.1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._last_dist = math.dist(self._agent_location, self._target_location)

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly place the circle 
        self._agent_location = self.np_random.integers(0, self.window_size-self.cursor_size*2, size=2, dtype=int)
        self._target_location = self.np_random.integers(0, self.window_size-self.target_size*2, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _in_circle(self):
        if self.target:
            return math.sqrt((self.target.centerx - self.cursor.centerx)**2 + (self.target.centery - self.cursor.centery)**2) < (self.target[2]/2 + self.cursor[2]/2)
        else:
            return (self._target_location[0]-self._agent_location[0])**2 + (self._target_location[1]-self._agent_location[1])^2 < (self.target_size/2 + self.cursor_size/2)

    def _get_obs(self):
        return {"class": self._last_class, "mav": self._last_mav, "probs": self._last_probs}

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
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont('helvetica', 40)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw Circle 
        self.cursor = pygame.draw.circle(canvas, self.RED, (self._target_location[0] + self.target_size, self._target_location[1] + self.target_size), self.target_size)
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
            pygame.display.update()

            # We want to wait for a new event from the EMG controller 
            data, _ = self.sock.recvfrom(1024)
            data = str(data.decode("utf-8"))
            if data:
                self._last_class = int(data.split(' ')[0])
                self._last_mavs = np.array([float(i) for i in data.split(' ')[6:]]) # MAVS 
                self._last_probs = np.array([float(i) for i in data.split(' ')[1:6]])

    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()