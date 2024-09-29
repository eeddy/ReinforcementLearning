import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces


class FittsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, discrete=True):
        self.window_size = 750 
        self.discrete = discrete

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
            }
        )

        if self.discrete:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self._action_to_direction = {
            0: np.array([20, 0]),
            1: np.array([0, 20]),
            2: np.array([-20, 0]),
            3: np.array([0, -20]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # Default parameters for ISO Fitts 
        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.cursor_size = 7
        self.target_size = 50

        self.cursor = None 
        self.target = None 

        self._last_dist = 1000000 

    def step(self, action):
        if self.discrete:
           direction = self._action_to_direction[int(action)] 
        else:
            direction = action * 15 # 15 is max speed 

        # Make sure the agent doesn't leave the screen 
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size - 10
        )

        terminated = self._in_circle()
        reward = 0 # Dont reward until final
        if terminated:
            reward = 1 
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
        return {"agent": self._agent_location, "target": self._target_location}

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
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw Circle 
        self.cursor = pygame.draw.circle(canvas, self.RED, (self._target_location[0] + self.target_size, self._target_location[1] + self.target_size), self.target_size)
        # Draw Cursor 
        self.target = pygame.draw.circle(canvas, self.BLACK, (self._agent_location[0] + self.cursor_size, self._agent_location[1] + self.cursor_size), self.cursor_size)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()