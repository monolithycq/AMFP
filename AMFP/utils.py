
ANTMAZE = [
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-large-diverse-v2",
    "antmaze-large-play-v2",
    #"antmaze-ultra-diverse-v0",
    #"antmaze-ultra-play-v0"
]

KITCHEN = ["kitchen-partial-v0", "kitchen-mixed-v0"]

GYM = [
    "halfcheetah-medium-v2",
    "walker2d-medium-v2",
    "hopper-medium-v2",
    "halfcheetah-medium-replay-v2",
    "walker2d-medium-replay-v2",
    "hopper-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "walker2d-medium-expert-v2",
    "hopper-medium-expert-v2"
]

TASKS = {'antmaze': ANTMAZE, 'kitchen': KITCHEN, 'gym': GYM}

def get_return_type(env_name: str):
    if env_name in GYM:
        return_type = 'rtg'
    else:
        return_type = 'state'
    return return_type

def check_env_name(env_name: str) -> bool:
    for name, task in TASKS.items():
        if env_name in set(task):
            return True
    return False