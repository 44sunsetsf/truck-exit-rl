"""Train a DQN policy to take a highway exit, as a passenger car or as a heavy truck.

The truck variant makes the ego vehicle 16.5 m long, caps its target speeds lower and
shapes the reward for heavy-duty behaviour: get onto the exit lane early, keep a larger
time gap and avoid needless lane changes.

Usage:
    python train.py --vehicle car --reward default --steps 50000
    python train.py --vehicle truck --reward default --steps 50000
    python train.py --vehicle truck --reward truck --steps 50000
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401  registers the environments
import numpy as np
from stable_baselines3 import DQN, PPO

TRUCK_LENGTH = 16.5


class TruckWrapper(gym.Wrapper):
    """Turns the ego car into a long, slower truck and optionally reshapes the reward."""

    def __init__(self, env, truck: bool, shaped: bool, crash_penalty: float = 1.0, gap_weight: float = 0.1):
        super().__init__(env)
        self.truck, self.shaped = truck, shaped
        self.crash_penalty, self.gap_weight = crash_penalty, gap_weight

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.truck:
            self.env.unwrapped.vehicle.LENGTH = TRUCK_LENGTH
        self.lane_changes = 0
        self.last_lane = self.env.unwrapped.vehicle.lane_index
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        lane = ego.lane_index
        changed = lane[2] != self.last_lane[2]
        self.lane_changes += int(changed)
        self.last_lane = lane
        gap = time_gap(self.env.unwrapped)
        info.update(time_gap=gap, lane_changes=self.lane_changes)
        if self.shaped:
            # early lane choice: reward being close to the exit side long before the exit
            road_lanes = self.env.unwrapped.config['lanes_count']
            reward += 0.05 * (lane[2] / max(road_lanes - 1, 1))
            # heavy vehicles need a larger time gap to the vehicle in front
            if gap < 2.5:
                reward -= self.gap_weight * (2.5 - gap) / 2.5
            # comfort: discourage lane changes that are not needed
            reward -= 0.02 * int(changed)
            if ego.crashed:
                reward -= self.crash_penalty
        return obs, reward, terminated, truncated, info


def time_gap(env) -> float:
    ego = env.vehicle
    front, _ = env.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None or ego.speed <= 0.1:
        return 10.0
    dist = front.position[0] - ego.position[0] - (front.LENGTH + ego.LENGTH) / 2
    return float(max(dist, 0.0) / ego.speed)


def make_env(vehicle: str, reward: str):
    env = gym.make('exit-v0')
    if vehicle == 'truck':
        env.unwrapped.configure({'action': {'type': 'DiscreteMetaAction', 'target_speeds': [14, 19, 24]}})
    # truck2: second shaping attempt with a much stronger crash penalty and gap weight
    if reward == 'truck2':
        return TruckWrapper(env, truck=vehicle == 'truck', shaped=True, crash_penalty=5.0, gap_weight=0.3)
    return TruckWrapper(env, truck=vehicle == 'truck', shaped=reward == 'truck')


def evaluate(model, vehicle: str, episodes: int, seed: int) -> dict:
    env = make_env(vehicle, 'default')
    exits = crashes = 0
    gaps, changes = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=10_000 + seed * 1000 + ep)
        done = False
        min_gap = 10.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            min_gap = min(min_gap, info['time_gap'])
            done = terminated or truncated
        exits += int(info.get('is_success', False))
        crashes += int(env.unwrapped.vehicle.crashed)
        gaps.append(min_gap)
        changes.append(info['lane_changes'])
    return {'episodes': episodes, 'exit_rate': exits / episodes, 'crash_rate': crashes / episodes,
            'mean_min_time_gap_s': float(np.mean(gaps)), 'mean_lane_changes': float(np.mean(changes))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle', choices=['car', 'truck'], default='car')
    parser.add_argument('--reward', choices=['default', 'truck', 'truck2'], default='default')
    parser.add_argument('--steps', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--algo', choices=['dqn', 'ppo'], default='dqn')
    args = parser.parse_args()

    name = f'{args.vehicle}_{args.reward}_s{args.seed}' + ('' if args.algo == 'dqn' else f'_{args.algo}')
    env = make_env(args.vehicle, args.reward)
    if args.algo == 'ppo':
        model = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4, n_steps=512,
                    batch_size=64, gamma=0.8, seed=args.seed, verbose=0)
    else:
        model = DQN('MlpPolicy', env, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=5e-4,
                    buffer_size=15_000, learning_starts=200, batch_size=32, gamma=0.8, train_freq=1,
                    gradient_steps=1, target_update_interval=50, exploration_fraction=0.7,
                    seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.steps)
    Path('models').mkdir(exist_ok=True)
    model.save(f'models/{name}')

    # every policy is evaluated on the truck, the vehicle we care about
    results = {'name': name, 'steps': args.steps, 'eval_on_truck': evaluate(model, 'truck', args.episodes, args.seed)}
    Path('results').mkdir(exist_ok=True)
    Path(f'results/{name}.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
