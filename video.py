import cv2
import imageio
import numpy as np
import wandb


class VideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            elif hasattr(env, 'get_pixels'):  # ARC 환경용
                frame = env.get_pixels()
                frame = cv2.resize(frame, (self.render_size, self.render_size),
                                interpolation=cv2.INTER_NEAREST)
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            if isinstance(obs, dict) and 'grid' in obs:  # ARC 환경용
                frame = self._render_arc_obs(obs)
            else:  # DMControl 환경용
                frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                            dsize=(self.render_size, self.render_size),
                            interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def _render_arc_obs(self, obs):
        grid = obs['grid'].reshape(30, 30)
        frame = self._grid_to_rgb(grid)
        return cv2.resize(frame, (self.render_size, self.render_size),
                        interpolation=cv2.INTER_NEAREST)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)

    def _grid_to_rgb(self, grid):
        colors = {
            0: [0, 0, 0],      # 검정
            1: [0, 0, 255],    # 파랑 
            2: [255, 0, 0],    # 빨강
            3: [0, 255, 0],    # 초록
            4: [255, 255, 0],  # 노랑
            5: [128, 128, 128],# 회색
            6: [255, 0, 255],  # 보라
            7: [255, 165, 0],  # 주황
            8: [0, 255, 255],  # 하늘
            9: [128, 0, 0]     # 갈색
        }
        H, W = grid.shape
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                frame[i,j] = colors[grid[i,j]]
        return frame