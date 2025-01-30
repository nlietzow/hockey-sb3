import numpy as np
from hockey.hockey_env import BasicOpponent, HockeyEnv
from sbx import CrossQ

from utils import CHECKPOINTS_DIR

if __name__ == '__main__':
    env = HockeyEnv()
    crossq = CrossQ.load(CHECKPOINTS_DIR / "cross_q" / "model")
    opponent = BasicOpponent(weak=False)

    for _ in range(100):
        obs, info = env.reset()
        done = False
        while not done:
            obs2 = env.obs_agent_two()
            a1, _ = crossq.predict(obs)
            a2, _ = opponent.predict(obs2)
            obs, _, done, _, info = env.step(np.hstack([a1, a2]))

        print(info["is_success"])
