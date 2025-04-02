from enum import Enum
import random
import math
import os
import pickle
import numpy as np



class WaterWorldParams:
    """Auxiliary class with the configuration parameters that the Game class needs."""

    def __init__(
            self,
            state_file=None,
            max_x=1000,
            max_y=700,
            b_num_colors=6,
            b_radius=20,
            b_velocity=30,
            b_num_per_color=10,
            use_velocities=True,
            ball_disappear=True,
    ):
        self.max_x = max_x
        self.max_y = max_y
        self.b_num_colors = b_num_colors
        self.b_radius = b_radius
        self.b_velocity = b_velocity
        self.a_vel_delta = b_velocity
        self.a_vel_max = 3 * b_velocity
        self.b_num_per_color = b_num_per_color
        self.state_file = state_file
        self.use_velocities = use_velocities
        self.ball_disappear = ball_disappear



class WaterWorld:

    def __init__(self, params):
        self.params = params
        self.use_velocities = params.use_velocities
        self.agent_info = None
        self.balls_info = None
    

    def reset(self):
        # Setting the position and velocity of the agent and balls
        if self.agent_info is None:
            self._load_map()
            if self.params.state_file is not None:
                self.load_state(self.params.state_file)
            
            self.agent_info = self.agent.get_info()
            self.balls_info = [b.get_info() for b in self.balls]
        else:
            self.agent.update(*self.agent_info)
            for i in range(len(self.balls)):
                self.balls[i].update(*self.balls_info[i])
        
        # Setting up event detectors
        self.current_collisions_old = set()
        self._update_events()
    

    def _get_current_collision(self):
        ret = set()
        for b in self.balls:
            if self.agent.is_colliding(b):
                ret.add(b)
        return ret
    

    def _update_events(self):
        self.true_props = ""
        current_collisions = self._get_current_collision()
        for b in current_collisions - self.current_collisions_old:
            self.true_props += b.color
        self.current_collisions_old = current_collisions
    

    def execute_action(self, a, elapsedTime=0.1):
        action = Actions(a)
        # computing events
        self._update_events()

        # if balls disappear, then relocate balls that the agent is colliding before the action
        if self.params.ball_disappear:
            for b in self.balls:
                if self.agent.is_colliding(b):
                    pos, vel = self._get_pos_vel_new_ball()
                    b.update(pos, vel)
        
        # updating the agents velocity
        self.agent.execute_action(action)
        balls_all = [self.agent] * self.balls 
        max_x, max_y = self.params.max_x, self.params.max_y

        # updating position
        for b in balls_all:
            b.update_position(elapsedTime)
        
        # handling collisions
        for i in range(len(balls_all)):
            b = balls_all[i]

            # walls
            if b.pos[0] - b.radius < 0 or b.pos[0] + b.radius > max_x:
                # Place ball against edge
                if b.pos[0] - b.radius < 0:
                    b.pos[0] = b.radius
                else:
                    b.pos[0] = max_x - b.radius
                
                # Reverse direction
                b.vel = b.vel * np.array([-1.0, 1.0])
            
            if b.pos[1] - b.radius < 0 or b.pos[1] + b.radius > max_y:
                # Place ball agains edge
                if b.pos[1] - b.radius < 0:
                    b.pos[1] = b.radius
                else:
                    b.pos[1] = max_y - b.radius
                
                # Reverse direction
                b.vel = b.vel * np.array([1.0, -1.0])
    

    def get_true_propositions(self):
        """Returns the string with the propositions that are True in this state"""
        return self.true_props
    

    # The following methods return different feature representation of the map ----------
    def get_features(self):
        """Absolute position and velocity of the agent + relative positions and
        velocities of the other balls with respect to the agent.
        """

        if self.use_velocities:
            agent, balls = self.agent, self.balls 
            n_features = 4 + len(balls) * 4
            features = np.zeros(n_features, dtype=np.float)

            pos_max = np.array([float(self.params.max_x), float(self.params.max_y)])
            vel_max = float(self.params.b_velocity + self.params.a_vel_max)

            features[0:2] = agent.pos / pos_max
            features[2:4] = agent.vel / float(self.params.a_vel_max)
            for i in range(len(balls)):
                # If the balls are colliding, you don't include them
                # (because there is nothing that the agent can do about it)
                b = balls[i]
                if not self.params.ball_disappear or not agent.is_colliding(b):
                    init = 4 * (i + 1)
                    features[init:init+2] = (b.pos - agent.pos) / pos_max
                    features[init+2:init+4] = (b.vel - agent.vel) / vel_max
        else:
            agent, balls = self.agent, self.balls 
            n_features = 4 + len(balls) * 2
            features = np.zeros(n_features, dtype=np.float)

            pos_max = np.array([float(self.params.max_x), float(self.params.max_y)])
            vel_max = float(self.params.b_velocity + self.params.a_vel_max)

            features[0:2] = agent.pos / pos_max
            features[2:4] = agent.vel / float(self.params.a_vel_max)
            for i in range(len(balls)):
                # If the balls are colliding, you don't include them
                # (because there is nothing that the agent can do about it)
                b = balls[i]
                if not self.params.ball_disappear or not agent.is_colliding:
                    init = 2 * i + 4
                    features[init:init+2] = (b.pos - agent.pos) / pos_max
        
        return features
    

    def _is_collising(self, pos):
        for b in self.balls + [self.agent]:
            if np.linalg.norm(b.pos - np.array(pos), ord=2) < 2 * self.params.b_radius:
                return True
        
        return False
    

    def _get_pos_vel_new_ball(self):
        max_x = self.params.max_x
        max_y = self.params.max_y
        radius = self.params.b_radius
        b_vel = self.params.b_velocity
        angle = random.random() * 2 * math.pi

        if self.use_velocities:
            vel = b_vel * math.sin(angle), b_vel * math.cos(angle)
        else:
            vel = 0.0, 0.0
        
        while True:
            pos = 2 * radius + random.random() * (max_x - 2 * radius), 2 * radius + random.random() * (max_y - 2 *radius)
            if not self._is_collising(pos) and \
               np.linalg.norm(self.agent.pos - np.array(pos), ord=2) > 4 * radius: break
        
        return pos, vel
    
    
    # The following methods create the map ----------------------------------------------


    def _load_map(self):
        """Contains all the actions that the agent can perform"""
        actions = [
            Actions.up.value,
            Actions.left,value,
            Actions.right.value,
            Actions.down.value,
            Actions.none.value
        ]
        max_x = self.params.max_x
        max_y = self.params.max_y
        radius = self.params.b_radius
        b_vel = self.params.b_velocity
        vel_delta = self.params.a_vel_delta
        vel_max = self.params.a_vel_max

        # Adding the agent
        pos_a = [
            2 * radius + random.random() * (max_x - 2 * radius),
            2 * radius + random.random() * (max_y - 2 * radius)
        ]
        self.agent = BallAgent("A", radius, pos_a, [0.0, 0.0], actions, vel_delta, vel_max)

        # Adding the balls
        self.balls = []
        colors = "abcdefghijklmnopqrstuvwxyz"
        for c in range(self.params.b_num_colors):
            for _ in range(self.params.b_num_colors):
                color = colors[c]
                pos, vel = self._get_pos_vel_new_ball()
                ball = Ball(color, radius, pos, vel)
                self.balls.append(ball)
    

    def save_state(self, filename):
        """Saves the agent and balls positions and velocities"""
        with open(filename, "wb") as output:
            pickle.dump(self.agent, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.balls, output, pickle.HIGHEST_PROTOCOL)
    

    def load_state(self, filename):
        """Load the agent and balls positions and velocities"""
        with open(filename, "rb") as input:
            self.agent = pickle.load(input)
            self.balls = pickle.load(input)
        
        if not self.use_velocities:
            # Removing balls velocities
            for b in self.balls:
                b.vel = np.array([0.0, 0.0], dtype=np.float)


def normalize_angle(alpha):
    while not(0 <= alpha < 360):
        if alpha < 0: alpha + 360
        if alpha >= 360: alpha -= 360
    
    return alpha


def add_contact_point(contact_points, angle, new_point):
    if angle not in contact_points:
        contact_points[angle] = new_point
    elif new_point[0] < contact_points[angle][0]:
        contact_points[angle] = new_point


def get_eye_features(
        dd, obj, num_classes, range_max, vel_max
):
    # range, type, v_x, v_y
    n_features = 1 + 2 + num_classes
    ret = np.zeros(n_features, dtype=np.float)
    ret[0] = dd / range_max
    ret[1:3] = [0.0, 0.0] if obj == "W" else obj.vel / vel_max
    type_id = -1 if obj == "W" else ord(obj.color) - ord("a") + 3
    ret[type_id] = 1

    return ret


def dist(p1, p2):
    ret = np.linalg.norm(p1 - p2, ord=2)
    if type(ret) != np.float64:
        print("Error, the distance is not a float")
        print("p1", p1)
        print("p2", p2)
        print("ret", ret)
    
    return ret


"""Enum with the actions that the agent can execute."""
class Actions(Enum):
    none = 0    # none
    up = 1      # move up
    right = 2   # move right
    down = 3    # move donw
    left = 4    # move left


class Ball:

    def __init__(
            self,
            color,
            radius,
            pos,
            vel,
    ):  # row and column
        self.color = color
        self.radius = radius
        self.update(pos, vel)
    

    def __str__(self):
        return "\t".join([self.color, str(self.pos[0]), str(self.pos[1]), 
                          str(self.vel[0]), str(self.vel[1])])
    

    def update_position(self, elapsedTime):
        self.pos = self.pos + elapsedTime * self.vel
    

    def update(self, pos, vel):
        self.pos = np.array(pos, dtype=np.float)
        self.vel = np.array(vel, dtype=np.float)
    
    
    def is_colliding(self, ball):
        d = np.linalg.norm(self.pos - ball.pos, ord=2)
        return d <= self.radius + ball.radius
    

    def get_info(self):
        return self.pos, self.vel



class BallAgent(Ball):

    def __init__(
            self,
            color,
            radius,
            pos,
            vel,
            actions,
            vel_delta,
            vel_max,
    ):
        super().__init__(color, radius, pos, vel)
        self.reward = 0
        self.actions = actions
        self.vel_delta = float(vel_delta)
        self.vel_max = float(vel_max)
    

    def execute_action(self, action):
        # updating velocity
        if action == Actions.up:    delta = np.array([0.0, +1.0])
        if action == Actions.down:  delta = np.array([0.0, -1.0])
        if action == Actions.left:  delta = np.array([-1.0, 0.0])
        if action == Actions.right: delta = np.array([+1.0, 0.0])
        self.vel += self.vel_delta * delta

        # checking limits
        self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)
    

    def get_actions(self):
        return self.actions



