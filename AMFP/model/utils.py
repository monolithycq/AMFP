import os
from torch import device
from typing import Tuple, Union
import numpy as np
from d4rl import offline_env
from d4rl.kitchen import kitchen_envs
from AMFP.data.sequence import RETURN_DIMS

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

Device = Union[device, str, int, None]


def get_return(env: offline_env.OfflineEnv, return_type: str = 'state', return_frac=None, obs=None):
    if return_type == 'state':
        if 'antmaze' in env.spec.id:
            _return = env.target_goal

        elif 'kitchen' in env.spec.id:
            _return = obs[:30].copy()
            subtask_collection = env.TASK_ELEMENTS
            for task in subtask_collection:
                subtask_indices = kitchen_envs.OBS_ELEMENT_INDICES[task]
                subtask_goals = kitchen_envs.OBS_ELEMENT_GOALS[task]
                _return[subtask_indices] = subtask_goals
            return_mask = np.ones(30, dtype=np.bool8)
            return_mask[RETURN_DIMS['kitchen']] = False
            _return = np.where(return_mask, 0., _return)
        else:
            raise NotImplementedError
    else:  # rtg as return
        max_score = env.ref_max_score / env._max_episode_steps
        min_score = env.ref_min_score / env._max_episode_steps
        _return = min_score + (max_score - min_score) * return_frac
        _return = np.array([_return], dtype=np.float32)

    return _return


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)