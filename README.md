# truck-exit-rl: teaching a car-style driving policy to take an exit like a heavy truck

**Status: work in progress (started October 2026). First results below; more runs are being added.**

A policy trained for passenger cars does not know that a long, heavy vehicle needs larger gaps and must move towards an
exit early. This project uses the `exit-v0` scenario of [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
as a small closed-loop testbed: the ego vehicle must cross several lanes of dense traffic to reach a mandatory exit.

## Setup

- **Vehicles:** the default passenger car, and a truck variant: 16.5 m long, with lower target speeds (14 / 19 / 24 m/s).
  highway-env has no articulated trailer, so off-tracking is out of scope here.
- **Rewards:** the default `exit-v0` reward, and a truck-shaped reward that adds
  - a small bonus for being on the exit side of the road early,
  - a penalty for time gaps under 2.5 s to the vehicle in front,
  - a small penalty per lane change (comfort),
  - an explicit crash penalty.
- **Policies:** DQN (Stable-Baselines3, MLP 256x256), 50k steps.
- **Evaluation:** every policy is driven as the truck on 100 held-out seeds and scored on exit success rate,
  crash rate, minimum time gap and number of lane changes.

## Runs

| Policy | Trained as | Reward |
|---|---|---|
| car baseline | car | default |
| truck, default reward | truck | default |
| truck, shaped reward | truck | truck-shaped (v1, v2) |

```bash
pip install -r requirements.txt
python train.py --vehicle car --reward default --steps 50000
python train.py --vehicle truck --reward default --steps 50000
python train.py --vehicle truck --reward truck --steps 50000
```

Next: more seeds, an ablation of each reward term, and PPO for comparison.

## Results

**First runs** (DQN, 50k steps, seed 0). Every policy is driven as the truck on 100 held-out seeds:

| Policy | Trained as | Reward | Exit success | Crash rate | Mean min. time gap | Lane changes |
|---|---|---|---|---|---|---|
| car baseline | car | default | 3% | 35% | 4.1 s | 1.8 |
| truck, default reward | truck | default | 12% | 85% | 1.9 s | 3.1 |
| truck, shaped v1 | truck | shaped (crash −1, gap weight 0.1) | 8% | 92% | 0.5 s | 2.9 |

What this shows so far:
- `exit-v0`'s default reward has **no collision cost** (`collision_reward: 0`), so a crash simply ends the episode.
  The policy trained on it as the truck goes for the exit aggressively and crashes in 85% of episodes.
- The car-trained policy is passive rather than safe: it crashes less but almost never reaches the exit.
- The first shaped reward was too weak to change that within 50k steps: the small per-step exit-side bonus outweighs
  the gap and crash terms. A second version with a crash penalty of −5 and a gap weight of 0.3 is training now.

These are single-seed, short runs; treat them as a first look, not a benchmark.
