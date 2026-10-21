# truck-exit-rl: teaching a car-style driving policy to take an exit like a heavy truck

**Status: complete enough for a thesis pitch (October 2026).** Early 50k-step runs diagnosed the reward;
main numbers below use longer training, a reward ablation, and a longer GRPO pass.

A policy trained for passenger cars does not know that a long, heavy vehicle needs larger gaps and must move towards an
exit early. This project uses the `exit-v0` scenario of [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
as a small closed-loop testbed: the ego vehicle must cross several lanes of dense traffic to reach a mandatory exit.

## Setup

- **Vehicles:** the default passenger car, and a truck variant: 16.5 m long, with lower target speeds (14 / 19 / 24 m/s).
  highway-env has no articulated trailer, so **off-tracking is out of scope** (honest gap vs a real tractor-semitrailer thesis).
- **Rewards:** the default `exit-v0` reward, and a truck-shaped reward that adds
  - a small bonus for being on the exit side of the road early,
  - a penalty for time gaps under 2.5 s to the vehicle in front,
  - a small penalty per lane change (comfort),
  - an explicit crash penalty.
- **Policies:** DQN and PPO (Stable-Baselines3, MLP 256×256); GRPO (TRL) on Qwen2.5-0.5B-Instruct for a text-scene
  action head (KEEP / RIGHT / LEFT / SLOW) as a small stand-in for LM-style post-training.
- **Evaluation:** every driving policy is driven **as the truck** on 100 held-out seeds; metrics are exit success,
  crash rate, mean minimum time gap, and lane changes.

## Run

```bash
pip install -r requirements.txt
python train.py --vehicle car --reward default --steps 200000
python train.py --vehicle truck --reward default --steps 200000
python train.py --vehicle truck --reward truck --steps 200000
python grpo/grpo_exit.py --steps 200 --temperature 1.5
```

## Results

### Phase A — diagnosis (DQN, 50k steps, seed 0)

| Policy | Trained as | Reward | Exit success | Crash rate | Mean min. gap | Lane changes |
|---|---|---|---|---|---|---|
| car baseline | car | default | 3% | 35% | 4.1 s | 1.8 |
| truck, default | truck | default | 12% | 85% | 1.9 s | 3.1 |
| truck, shaped v1 | truck | crash −1, gap 0.1 | 8% | 92% | 0.5 s | 2.9 |
| truck, shaped v2 | truck | crash −5, gap 0.3 | 19% | 77% | 1.0 s | 3.8 |
| PPO, default | truck | default | 0% | 4% | 6.9 s | 1.8 |
| PPO, shaped v2 | truck | shaped v2 | 0% | 0% | 10 s | 0.0 |

`exit-v0`'s default reward has **no collision cost** (`collision_reward: 0`). Trained as the truck, DQN goes for the
exit and crashes in 85% of episodes. Shaped v1 was too weak; v2 (crash −5, gap weight 0.3) moved exit 12% → 19% and
crashes 85% → 77%. PPO stayed passive (safe, never exits) within 50k steps.

### Phase B — longer DQN (200k steps, seeds 0/1/2, mean ± std, eval as truck)

| Policy | Reward | Exit success | Crash rate | Mean min. gap |
|---|---|---|---|---|
| truck, default | default | 14% ± 3 | 81% ± 4 | 2.0 s ± 0.3 |
| truck, shaped v2 | shaped v2 | **34% ± 4** | **51% ± 5** | 1.8 s ± 0.2 |
| car baseline (eval as truck) | default | 4% ± 1 | 33% ± 3 | 4.0 s ± 0.4 |
| PPO, shaped v2 | shaped v2 | 9% ± 3 | 18% ± 4 | 5.1 s ± 0.6 |

Longer training makes shaped v2 clearly better than the default truck reward, and better than the car policy evaluated
as a truck. PPO starts to find the exit occasionally but stays conservative.

### Phase C — reward ablation (DQN shaped v2, 200k, seed 0)

| Ablation | Exit success | Crash rate | Note |
|---|---|---|---|
| Full shaped v2 | 36% | 49% | reference |
| No crash penalty | 28% | 74% | exit-seeking without fear |
| No gap penalty | 31% | 58% | closer following, more scrapes |
| No exit-side bonus | 22% | 47% | waits too long to commit |
| No lane-change cost | 33% | 55% | slightly more weaving |

Crash and exit-side terms matter most for this scenario; the comfort term is secondary.

### Phase D — GRPO on a small LM (text scenes → discrete actions)

Rule-based heavy-truck reward on KEEP / RIGHT / LEFT / SLOW. Held-out: 200 scenes.

| Run | Steps | Temp | Agreement with rule |
|---|---|---|---|
| First look | 30 | 1.0 | 27% → 27% (groups often identical → zero advantage) |
| Diversity fix | 15 | 1.5 | 27% → 24% (noise; but zero-spread groups fall to 0–5%) |
| Longer pass | 200 | 1.5 | **27% → 41%** |

Temperature 1.5 restores within-group reward spread; with enough steps GRPO moves agreement up by about 14 points.
This is still a **text** stand-in, not a vision-language-action model — useful for the post-training loop, not a
substitute for Alpamayo.

## Limitations

- No articulated trailer / off-tracking (simulator limit).
- Discrete actions and a small MLP policy; not a 10B VLA.
- Dense traffic is synthetic; numbers are for method comparison, not fleet KPI.

## Takeaway

Reward design dominates: measure the car baseline in closed loop, shape for early exit commitment and gaps, ablate
each term, then spend budget on post-training (DQN length, then GRPO-style updates) only after the reward is not lying.
