import os
import gym
import d4rl

import collections
import numpy as np
import pickle

from d4rl import offline_env
from AMFP.utils import TASKS,GYM


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DATA_DIR, "datasets")


def get_average_rtgs(rewards, max_episode_steps):
    average_rtgs = np.zeros_like(rewards)
    for i in range(len(rewards)):
        average_rtgs[i] = np.sum(rewards[i:]) / (max_episode_steps - i)
    return average_rtgs


def parse_episode(env: offline_env.OfflineEnv):
    '''
    D4RL store data as one continuous trajectory
    Split it into episodes/trajectories with "terminals" or "timeouts" flag
    '''
    dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    cur_episode = collections.defaultdict(list)

    use_timeouts = "timeouts" in dataset

    episode_step = 0
    episodes = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        final_timestep = dataset["timeouts"][i] if use_timeouts else episode_step == env._max_episode_steps - 1  # timeout stands for reaching max steps

        if 'antmaze' in env.spec.id:
            done = final_timestep
        else:
            done = done_bool or final_timestep

        for key in dataset:
            if 'metadata' in key: continue
            cur_episode[key].append(dataset[key][i])

        if done:
            episode_step = 0
            if env.spec.id in GYM:  # get avg return like RvS-R, only for gym
                cur_episode['avg_rtgs'] = get_average_rtgs(cur_episode['rewards'], env._max_episode_steps)
            for key in cur_episode:
                cur_episode[key] = np.array(cur_episode[key])
                if cur_episode[key].ndim == 1:  # make sure all keys have ndim == 2
                    cur_episode[key] = np.expand_dims(cur_episode[key], axis=-1)
            episodes.append(cur_episode)
            cur_episode = collections.defaultdict(list)

        episode_step += 1

    return episodes


def parse_pickle_datasets(env_name: str, output_dir: str):
    env = gym.make(env_name)
    episodes = parse_episode(env)
    output_path = f'{output_dir}/{env_name}.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(episodes, f)


if __name__ == '__main__':
    # print(gym.envs.registry.all())
    os.makedirs(DATASET_DIR, exist_ok=True)
    print('Start parsing and pickling data')
    for name, task in TASKS.items():
        for env_name in task:
            parse_pickle_datasets(env_name, DATASET_DIR)
    print('Pickle finished')