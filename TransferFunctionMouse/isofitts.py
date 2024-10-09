import pygame
import math
import time
import pickle
import os
import socket
import numpy as np
import random
from stable_baselines3 import PPO

class FittsLawTest:
    def __init__(self, num_circles=30, num_trials=15, savefile="out.pkl", logging=True, width=1250, height=750, small_rad=50, big_rad=50, vel=1, transfer_function=0):
        pygame.init()
        self.font = pygame.font.SysFont('helvetica', 40)
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        pygame.mouse.set_visible(False) 
        
        # logging information
        self.log_dictionary = {
            'trial_number':      [],
            'goal_circle' :      [],
            'global_clock' :     [],
            'cursor_position':   [],
            'current_direction': []
        }

        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.YELLOW = (255,255,0)
        self.BLUE   = (0,102,204)
        self.small_rad = small_rad
        self.big_rad   = big_rad
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.done = False
        self.dwell_time = 0.5
        self.num_of_circles = num_circles 
        self.max_trial = num_trials
        self.width = width
        self.height = height
        self.fps = 60
        self.duration = 0
        self.savefile = savefile
        self.logging = logging
        self.trial = 0
        self.cursor_size = 14
        self.vel = vel 
        self.dwell_timer = None
        self.acquired_target = False
        self.transfer_function = transfer_function # 0 = normal mouse, 1 = constant, 2 = RL 

        self.time_since_render = 0
        self.clock = pygame.time.Clock()

        # interface objects
        self.circles = []
        self.cursor = pygame.Rect(self.width//2 - 7, self.height//2 - 7, self.cursor_size, self.cursor_size)
        self.goal_circle = -1
        self.get_new_goal_circle()

        # Socket for reading EMG
        self.modelx = np.load('transfer_func_x.npy')
        self.modely = np.load('transfer_func_y.npy')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1/self.fps) 
        self.sock.bind(('127.0.0.1', 12345))

    def draw(self):
        self.screen.fill(self.BLACK)
        self.draw_circles()
        self.draw_cursor()
        self.draw_timer()
    
    def draw_circles(self):
        if not len(self.circles):
            self.angle = 0
            self.angle_increment = 360 // self.num_of_circles
            while self.angle < 360:
                self.circles.append(pygame.Rect((self.width//2 - self.small_rad) + math.cos(math.radians(self.angle)) * self.big_rad, (self.height//2 - self.small_rad) + math.sin(math.radians(self.angle)) * self.big_rad, self.small_rad * 2, self.small_rad * 2))
                self.angle += self.angle_increment

        for circle in self.circles:
            pygame.draw.circle(self.screen, self.RED, (circle.x + self.small_rad, circle.y + self.small_rad), self.small_rad, 2)
        
        goal_circle = self.circles[self.goal_circle]
        pygame.draw.circle(self.screen, self.RED, (goal_circle.x + self.small_rad, goal_circle.y + self.small_rad), self.small_rad)
            
    def draw_cursor(self):
        pygame.draw.circle(self.screen, self.YELLOW, (self.cursor.x + 7, self.cursor.y + 7), 7)

    def draw_timer(self):
        if hasattr(self, 'dwell_timer'):
            if self.dwell_timer is not None:
                toc = time.time()
                duration = round((toc-self.dwell_timer),2)
                time_str = str(duration)
                draw_text = self.font.render(time_str, 1, self.BLUE)
                self.screen.blit(draw_text, (10, 10))

    def update_game(self):
        self.draw()
        self.run_game_process()
    
    def run_game_process(self):
        self.check_collisions()
        self.check_events()

    def check_collisions(self):
        circle = self.circles[self.goal_circle]
        if math.sqrt((circle.centerx - self.cursor.centerx)**2 + (circle.centery - self.cursor.centery)**2) < (circle[2]/2 + self.cursor[2]/2):
            return True
        return False

    def check_events(self):
        # closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                return
        
        if self.transfer_function == 0:
            mouse = pygame.mouse.get_pos()
            self.cursor.x = mouse[0]
            self.cursor.y = mouse[1]
        
        if self.check_collisions():
            if self.dwell_timer is None:
                self.dwell_timer = time.time()
            else:
                toc = time.time()
                self.duration = round((toc - self.dwell_timer), 2)

            if self.duration >= self.dwell_time:
                self.get_new_goal_circle()
                self.dwell_timer = None
                if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                    self.trial += 1
                else:
                    if self.logging:
                        self.save_log()
                    self.done = True
                self.duration = 0
        else:
            self.duration = 0
            self.dwell_timer = None
            

    def move(self, x=None, y=None):
        if x == None and y == None:
            try:
                data, _ = self.sock.recvfrom(1024)
            except:
                return 

            data = str(data.decode("utf-8"))
            if data:
                x = int(data.split(' ')[0])
                y = int(data.split(' ')[1])

            if self.transfer_function == 2:
                x = self.modelx[np.abs(x)] * x 
                y = self.modely[np.abs(y)] * y 

        # Making sure its within the bounds of the screen
        if self.cursor.x + x > 0 + self.cursor_size//2 and self.cursor.x + x + self.cursor_size//2 < self.width:
            self.cursor.x += x 
        if self.cursor.y + y  > 0 + self.cursor_size//2 and self.cursor.y + y + self.cursor_size//2 < self.height:
            self.cursor.y += y

        
    def get_new_goal_circle(self):
        if self.goal_circle == -1:
            self.goal_circle = 0
            self.next_circle_in = self.num_of_circles//2
            self.circle_jump = 0
        else:
            self.goal_circle =  (self.goal_circle + self.next_circle_in )% self.num_of_circles
            if self.circle_jump == 0:
                self.next_circle_in = self.num_of_circles//2 + 1
                self.circle_jump = 1
            else:
                self.next_circle_in = self.num_of_circles // 2
                self.circle_jump = 0

    def log(self):
        circle = self.circles[self.goal_circle]
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_circle'].append((circle.centerx, circle.centery, circle[2]))
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.centerx, self.cursor.centery, self.cursor[2]))
        self.log_dictionary['current_direction'].append(self.cursor)

    def save_log(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        # Adding timestamp
        with open('results/'+ self.savefile, 'wb') as f:
            pickle.dump(self.log_dictionary, f)

    def run(self):
        while not self.done:
            if self.transfer_function == 1 or self.transfer_function == 2:
                self.move()
            self.update_game()
            # Render Screen ---> Something wonky going on here that I need to figure out .
            if time.time() - self.time_since_render >= 1/self.fps:
                self.log()
                pygame.display.update()
                self.time_since_render = time.time()
            if self.transfer_function == 0:
                self.clock.tick(self.fps)
        self.sock.close()
        pygame.quit()


IDS = [
    [50, 20],
    [150, 30],
    [310, 10],
]

tfs = [1,2]
random.shuffle(tfs)
for tf in tfs:
    for i, id in enumerate(IDS):
        FittsLawTest(big_rad=id[0], small_rad=id[1], savefile='t_' + str(tf) + '_' + str(i) + '.pkl', num_circles=8, transfer_function=tf).run()