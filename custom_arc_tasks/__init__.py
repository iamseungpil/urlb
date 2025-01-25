from .arc_env import ARCToDMCAdapter

# custom_arc_tasks/__init__.py
def make(task, obs_type='states', seed=None):
    return ARCToDMCAdapter()