# truck-exit-rl: teaching a car-style driving policy to take an exit like a heavy truck

**Status: work in progress (started October 2026). Results will be added here as runs finish.**

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
| truck, shaped reward | truck | truck-shaped |

```bash
pip install -r requirements.txt
python train.py --vehicle car --reward default --steps 50000
python train.py --vehicle truck --reward default --steps 50000
python train.py --vehicle truck --reward truck --steps 50000
```

Next: more seeds, an ablation of each reward term, and PPO for comparison.

## Results

Coming soon.
