#Author: Kazi Sifatul Islam

import os
import numpy as np
import pygame
from pygame.locals import *
from random import randrange
from math import sqrt, sin, cos, pi

import gym
from gym import spaces

import Constants
from Agent import Drone

class droneEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, render_mode="none"):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- ENV CONSTANTS ---
        self.WIDTH = Constants.WIDTH
        self.HEIGHT = Constants.HEIGHT
        self.gravity = Constants.gravity
        self.FPS = Constants.FPS
        self.thruster_mean = Constants.thruster_mean
        self.thruster_amp = Constants.thruster_amplitude
        self.diff_amp = Constants.diff_amplitude
        self.mass = Constants.mass
        self.arm = Constants.arm
        self.time_limit = Constants.TIME_LIMIT

        # --- AGENT & TARGET ---
        self.Agent = Drone()
        self.background = Constants.BACKGROUND
        self.spriter = Constants.spriter

        # --- OBS / ACTION SPACE ---
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.time = 0
        self.reward = 0
        self.target_counter = 0
        self.pace = 0

    # -----------------------------------
    # RESET
    # -----------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
    
        self.Agent.reset()
    
        self.x_target = randrange(50, self.WIDTH - 50)
        self.y_target = randrange(75, self.HEIGHT - 75)
    
        self.time = 0
        self.reward = 0
        self.target_counter = 0
    
        if self.render_mode == "human":
            self._init_pygame()
    
        return self._get_obs().astype(np.float32), {}

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        else:
            raise NotImplementedError("Render mode not supported.")


    # -----------------------------------
    # OBSERVATION CALCULATION
    # -----------------------------------
    def _get_obs(self):
        angle_to_up = self.Agent.angle * pi / 180
        velocity = sqrt(self.Agent.x_speed**2 + self.Agent.y_speed**2)
        angle_velocity = self.Agent.angle_speed
        angle_to_target = np.arctan2(
            self.y_target - self.Agent.y_position,
            self.x_target - self.Agent.x_position
        )
        angle_target_and_velocity = angle_to_target - np.arctan2(
            self.Agent.y_speed, self.Agent.x_speed + 1e-6
        )
        distance_to_target = sqrt(
            (self.x_target - self.Agent.x_position)**2 +
            (self.y_target - self.Agent.y_position)**2
        ) / 500
    
        obs = np.array([
            angle_to_up,
            velocity,
            angle_velocity,
            distance_to_target,
            angle_to_target,
            angle_target_and_velocity,
        ], dtype=np.float32)
    
        # CRITICAL FIX: Remove NaN / inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    
        return obs


    # -----------------------------------
    # STEP
    # -----------------------------------
    def step(self, action):
        self.pace = (self.pace + 1) % 20
        self.time += 1 / 60
        self.reward = 0.0

        action = int(action)

        # --- Physics update ---
        thr_left = self.thruster_mean
        thr_right = self.thruster_mean

        if action == 1:  # Up
            thr_left += self.thruster_amp
            thr_right += self.thruster_amp
        elif action == 2:  # Down
            thr_left -= self.thruster_amp
            thr_right -= self.thruster_amp
        elif action == 3:  # Right rotate
            thr_left += self.diff_amp
            thr_right -= self.diff_amp
        elif action == 4:  # Left rotate
            thr_left -= self.diff_amp
            thr_right += self.diff_amp

        # --- Forces ---
        self.Agent.angular_acceleration = self.arm * (thr_right - thr_left) / self.mass
        self.Agent.x_acceleration = -(thr_left + thr_right) * sin(self.Agent.angle * pi / 180) / self.mass
        self.Agent.y_acceleration = self.gravity - (thr_left + thr_right) * cos(self.Agent.angle * pi / 180) / self.mass

        # --- Apply physics ---
        self.Agent.x_speed += self.Agent.x_acceleration
        self.Agent.y_speed += self.Agent.y_acceleration
        self.Agent.angle_speed += self.Agent.angular_acceleration

        self.Agent.x_position += self.Agent.x_speed
        self.Agent.y_position += self.Agent.y_speed
        self.Agent.angle += self.Agent.angle_speed

        # --- Calculate reward ---
        dist = sqrt((self.Agent.x_position - self.x_target)**2 + (self.Agent.y_position - self.y_target)**2)

        self.reward += 1 / 60                       # survive bonus
        self.reward -= dist * 0.000166              # distance penalty

        terminated = False
        truncated = False

        # --- Collect target ---
        if dist < 45:
            self.reward += 100
            self.target_counter += 1
            self.x_target = randrange(50, self.WIDTH - 50)
            self.y_target = randrange(75, self.HEIGHT - 75)

        # --- Time limit ---
        if self.time > self.time_limit:
            truncated = True

        # --- Out of bounds ---
        if (
            self.Agent.x_position < -50 or self.Agent.x_position > self.WIDTH + 50 or
            self.Agent.y_position < -50 or self.Agent.y_position > self.HEIGHT + 50
        ):
            self.reward -= 800
            terminated = True

        # --- Render if needed ---
        if self.render_mode == "human":
            self._render_pygame()

        return self._get_obs(), self.reward, terminated, truncated, {}



    # -----------------------------------
    # PYGAME FUNCTIONS
    # -----------------------------------
    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.bg = pygame.transform.scale(
            pygame.image.load(self.background), 
            (self.WIDTH, self.HEIGHT)
        )
        self.Agent_img = self.spriter("Drone")
        self.Target_img = self.spriter("Baloon")
        self.font = pygame.font.SysFont("Arial", 20)

    def _render_pygame(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        self.screen.blit(self.bg, (0, 0))

        # Draw target
        t_img = self.Target_img[int(self.pace * 0.15) % len(self.Target_img)]
        self.screen.blit(t_img, (self.x_target - t_img.get_width()//2,
                                 self.y_target - t_img.get_height()//2))

        # Draw agent
        a_img = self.Agent_img[int(self.pace * 0.1) % len(self.Agent_img)]
        rotated = pygame.transform.rotate(a_img, self.Agent.angle)
        rect = rotated.get_rect(center=(self.Agent.x_position, self.Agent.y_position))
        self.screen.blit(rotated, rect)

        text = self.font.render(f"Targets: {self.target_counter}", True, (0, 0, 0))
        self.screen.blit(text, (20, 20))

        pygame.display.update()
        self.clock.tick(self.FPS)

    def close(self):
        if self.screen:
            pygame.quit()
