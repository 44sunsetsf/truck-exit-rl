"""Post-train a small language model with GRPO to choose truck-safe actions before a highway exit.

Each prompt describes an exit-ramp scene in text (distance to the exit, current lane, time gap to the
vehicle in front and in the target lane). The model answers with one action. A rule-based reward
encodes heavy-truck behaviour: move right early enough, never change lanes into a gap under 2.5 s,
slow down when the gap ahead is short. Accuracy against the rule is measured before and after.

Usage:
    python grpo/grpo_exit.py --steps 150
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
ACTIONS = ['KEEP', 'RIGHT', 'LEFT', 'SLOW']
MIN_GAP = 2.5


def make_scene(rng: random.Random) -> dict:
    lanes = rng.randint(3, 5)
    return {
        'lanes': lanes,
        'lane': rng.randint(1, lanes),  # 1 = leftmost, lanes = rightmost (exit side)
        'dist': rng.choice([200, 400, 600, 900, 1200]),
        'gap_ahead': round(rng.uniform(0.8, 5.0), 1),
        'gap_right': round(rng.uniform(0.8, 5.0), 1),
    }


def best_action(s: dict) -> str:
    lanes_to_go = s['lanes'] - s['lane']
    if s['gap_ahead'] < MIN_GAP - 1.0:
        return 'SLOW'
    # a truck needs roughly 300 m per lane change in dense traffic
    if lanes_to_go > 0 and s['dist'] <= 300 * (lanes_to_go + 1):
        return 'RIGHT' if s['gap_right'] >= MIN_GAP else 'SLOW'
    if s['gap_ahead'] < MIN_GAP:
        return 'SLOW'
    return 'KEEP'


def prompt(s: dict) -> str:
    return (
        'You drive a 40-tonne truck on a highway and must take the exit on the right.\n'
        f"Lanes: {s['lanes']} (lane 1 is leftmost, lane {s['lanes']} is next to the exit). You are in lane {s['lane']}.\n"
        f"Distance to the exit: {s['dist']} m. Time gap to the vehicle ahead: {s['gap_ahead']} s. "
        f"Time gap in the lane to your right: {s['gap_right']} s.\n"
        f'Answer with one word: {", ".join(ACTIONS)}.'
    )


def parse(text: str) -> str | None:
    m = re.search(r'\b(KEEP|RIGHT|LEFT|SLOW)\b', text.upper())
    return m.group(1) if m else None


def build(n: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = make_scene(rng)
        rows.append({'prompt': [{'role': 'user', 'content': prompt(s)}], 'answer': best_action(s)})
    return Dataset.from_list(rows)


def reward_fn(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        pred = parse(c[0]['content'])
        if pred is None:
            out.append(-1.0)
        elif pred == a:
            out.append(1.0)
        elif a == 'SLOW' or pred == 'LEFT':
            out.append(-0.5)  # unsafe: ignored a short gap, or moved away from the exit
        else:
            out.append(0.0)
    return out


@torch.no_grad()
def accuracy(model, tok, data: Dataset) -> dict:
    model.eval()
    correct = parsed = 0
    for row in data:
        ids = tok.apply_chat_template(row['prompt'], add_generation_prompt=True, return_tensors='pt',
                                      return_dict=True)['input_ids'].to(model.device)
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
        pred = parse(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
        parsed += pred is not None
        correct += pred == row['answer']
    return {'accuracy': correct / len(data), 'valid_answers': parsed / len(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--test', type=int, default=200)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    train, test = build(2000, seed=0), build(args.test, seed=1)

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to('mps')
    before = accuracy(base, tok, test)
    print('before', before, flush=True)

    cfg = GRPOConfig(output_dir='grpo/out', max_steps=args.steps, per_device_train_batch_size=16,
                     num_generations=8, max_completion_length=8, learning_rate=1e-5, beta=0.0, temperature=1.5,
                     logging_steps=10, save_strategy='no', report_to=[], seed=0)
    trainer = GRPOTrainer(model=base, reward_funcs=reward_fn, args=cfg, train_dataset=train, processing_class=tok)
    trainer.train()
    after = accuracy(trainer.model, tok, test)
    print('after', after, flush=True)

    Path('results').mkdir(exist_ok=True)
    Path(f'results/grpo_exit_{args.steps}steps_t15.json').write_text(json.dumps(
        {'model': MODEL, 'steps': args.steps, 'test_scenes': args.test, 'before': before, 'after': after}, indent=2))


if __name__ == '__main__':
    main()
