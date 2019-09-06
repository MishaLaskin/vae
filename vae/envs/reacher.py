import numpy as np
import pickle
import torch
from dm_control import suite
from gym.spaces import Box, Dict
from vae.models import VAE


class GoalImageEnv:

    def __init__(self,
                 env_name=None,
                 mode=None,
                 act_dim=None,
                 reward_type='pixeldiff',
                 img_dim=32,
                 camera_id=0,
                 path_length=200,
                 threshold=0.8,
                 gpu_id=0,
                 **kwargs):

        self.dm_env = suite.load(env_name, mode)
        self.task = self.dm_env._task
        self.camera_id = camera_id
        self.physics = self.dm_env.physics
        self.max_steps = path_length
        self.threshold = threshold
        self.reward_type = reward_type
        self.device = torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

        # actions always between -1.0 and 1.0
        self.action_space = Box(
            high=1.0, low=-1.0,
            shape=self.dm_env.action_spec().shape if act_dim is None else (act_dim,)
        )

        # observtions are, in principle, unbounded
        state_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(128,))

        goal_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(128,))

        self.observation_space = Dict([
            ('observation', state_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            #('state_observation', state_space),
            #('state_desired_goal', goal_space),
            #('state_achieved_goal', goal_space),
        ])

        self.render_kwargs = dict(
            width=img_dim, height=img_dim, camera_id=self.camera_id)

        self.model = VAE(
            img_dim=img_dim, image_channels=3, z_dim=128, device=self.device)

        self.model = self.model.to(self.device)

    def render(self):
        # renders image
        # example: self.render_kwargs={width=32,height=32,camera_id=0}
        return self.physics.render(**self.render_kwargs).astype(np.float32)

    def normalized_render(self):
        img = self.render()

        img = normalize_image(img)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = torch.tensor(img).float().to(self.device)
        _, z, _, _ = self.model(img)

        return z.squeeze(0).detach().cpu().numpy()

    def reset(self):
        self.steps = 0
        self.dm_env.reset()
        self.desired_goal = self.normalized_render()
        self.desired_goal
        self.dm_env.reset()
        obs = self.normalized_render()
        achieved_goal = obs.copy()
        obs_dict = dict(observation=obs,
                        achieved_goal=achieved_goal,
                        desired_goal=self.desired_goal)

        return obs_dict

    def step(self, a):
        # one timestep forward.astype(np.float32)
        # reward and done are taken from dm_control's env
        self.dm_env.step(a)
        obs = self.normalized_render()
        achieved_goal = obs.copy()

        obs_dict = dict(observation=obs,
                        achieved_goal=achieved_goal,
                        desired_goal=self.desired_goal)

        reward = self.compute_reward(a, obs_dict)
        done, is_success = self.is_done(reward)
        info = {
            'is_success': is_success
        }
        # self.update_internal_state()
        self.steps += 1
        return obs_dict, reward, done, info

    def compute_reward(self, action, obs, *args, **kwargs):
        # abstract method only cares about obs and threshold

        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

        if self.reward_type == 'threshold':
            r = -1.0 if distance > self.threshold else 0.0
        elif self.reward_type == 'pixeldiff':
            r = 0.0 if np.allclose(
                obs['achieved_goal'], obs['desired_goal']) else -1.0

        return r

    def compute_rewards(self, actions, obs):
        # abstract method only cares about obs and threshold
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']

        if self.reward_type == 'threshold':
            distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
            r = -(distances > self.threshold).astype(float)
        elif self.reward_type == 'pixeldiff':
            proximities = np.array([0.0 if np.allclose(x, y, atol=1e-3) else -1.0
                                    for x, y in zip(achieved_goals, desired_goals)])
            r = proximities

        return r

    def is_done(self, r):
        # abstract method only cares about obs and threshold
        # check if max step limit is reached
        if self.steps >= self.max_steps:
            done = True
            is_success = False
            return done, is_success

        # check if episode was successful
        is_success = r == 0
        done = is_success

        return done, is_success


def normalize_image(img):
    """normalizes image to [-1,1] interval

    Arguments:
        img {np.array or torch.tensor} -- [an image array / tensor with integer values 0-255]

    Returns:
        [np.array or torch tensor] -- [an image array / tensor with float values in [-1,1] interval]
    """
    # takes to [0,1] interval
    img /= 255.0
    # takes to [-0.5,0.5] interval
    # img -= 0.5
    # takes to [-1,1] interval
    # img /= 0.5
    return img


class Reacher(GoalImageEnv):

    def __init__(self,
                 path_length=None,
                 reward_type='threshold',
                 threshold=0.8,
                 img_dim=32,
                 camera_id=0):

        super().__init__(env_name='reacher',
                         mode='no_target',
                         act_dim=None,
                         reward_type=reward_type,
                         img_dim=img_dim,
                         camera_id=camera_id,
                         path_length=path_length)

        path = '/home/misha/research/baselines/vae/saved_models/reacher_vae.pth'

        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)
