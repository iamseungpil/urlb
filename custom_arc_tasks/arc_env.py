import dm_env
from dm_env import specs
from collections import OrderedDict
import numpy as np
from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader
from dm_env import StepType

class ARCToDMCAdapter(dm_env.Environment):
    def __init__(self, env=None, max_episode_steps=20):
        self._env = env or CustomArcEnv(
            data_loader=ARCLoader(),
            max_grid_size=(30, 30),
            colors=10
        )
        self._max_episode_steps = max_episode_steps
        self._convert_specs()

    def _convert_specs(self):
        # observation spec은 그대로 유지
        self._obs_spec = OrderedDict({
            'observations': specs.Array(
                shape=(30*30,),
                dtype=np.int32,
                name='observations'
            )
        })

        state, _ = self._env.reset()  # 현재 input grid 크기 얻기
        input_h, input_w = state['grid'].shape
        
        self._action_spec = (
            specs.DiscreteArray(input_h * input_w),  # position selection
            specs.DiscreteArray(10)  # color operation
        )

    def step(self, action):
        position, op = action
        current_grid = self._env.current_state['grid']
        input_h, input_w = current_grid.shape
        
        x = position // input_w
        y = position % input_w
        
        selection = np.zeros((30, 30), dtype=np.uint8)
        selection[x, y] = 1
        arc_action = {'selection': selection, 'operation': op}

        # 환경 스텝
        obs, reward, done, truncated, infos = self._env.step(arc_action)

        if infos.get('steps', 0) >= self._max_episode_steps:
            done = True
        
        # DMControl 형식으로 변환
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return dm_env.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=1.0 if not done else 0.0,
            observation={'observations': np.ravel(obs['grid'])}
        )

    def reset(self):
        state, info = self._env.reset()
        return dm_env.TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            discount=None,
            observation={'observations': state['grid'].ravel()}
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render_rgb(self):
        H, W = 30, 30
        img = np.zeros((H, W, 3), dtype=np.uint8)
        grid = self._env.current_state['grid']
        sel = self._env.current_state['selected']
        
        # ANSI 색상 매핑
        colors = {
            0: [0, 0, 0],      # 검정 (0)
            1: [0, 0, 255],    # 파랑 (12)
            2: [255, 0, 0],    # 빨강 (9)
            3: [0, 255, 0],    # 초록 (10)
            4: [255, 255, 0],  # 노랑 (11)
            5: [128, 128, 128],# 회색 (8)
            6: [255, 0, 255],  # 보라 (13)
            7: [255, 165, 0],  # 주황 (208)
            8: [0, 255, 255],  # 하늘 (14)
            9: [128, 0, 0]     # 갈색 (52)
        }
        
        for i in range(H):
            for j in range(W):
                color = colors[grid[i,j]]
                if sel[i,j]:
                    color = [min(c + 50, 255) for c in color]
                img[i,j] = color
                
        return img

    def get_pixels(self):
        return self.render_rgb()


class CustomArcEnv(O2ARCv2Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        # input grid size로 제한
        input_h, input_w = self.input.shape
        selection = action['selection'][:input_h, :input_w]
        action['selection'] = np.pad(selection, ((0, 30-input_h), (0, 30-input_w)))
        
        return super().step(action)

    def pick(self, data_index=None):
        data = self.loader.data[data_index or 0]
        return (
            data['train'][0]['input'],
            data['train'][0]['output'],
            data['test'][0]['input'],
            data['test'][0]['output'],
            ""
        )