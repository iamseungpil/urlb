DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'arc'  # ARC 도메인 추가
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

ARC_TASKS = [
    'arc_point',
    'arc_bbox',
    'arc_entire'
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + ARC_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'arc': 'arc_point'
}
