# Coup RL Agent

## Table of Contents

1. [Overview](#overview)
2. [Game Rules Summary](#game-rules-summary)

   * [Setup](#setup)
   * [Actions](#actions)
   * [Bluffing Mechanics](#bluffing-mechanics)
   * [Challenges vs Blocks](#challenges-vs-blocks)
   * [Steal Blocks](#steal-blocks)
3. [Project Architecture](#project-architecture)

   * [Environment Design](#environment-design)
   * [State & Observation Space](#state--observation-space)
   * [Action Space](#action-space)
   * [Reward Shaping](#reward-shaping)
   * [Algorithm Choice](#algorithm-choice)
   * [Compute Considerations](#compute-considerations)
4. [Evaluation Metrics](#evaluation-metrics)

   * [Baselines & Checkpoints](#baselines--checkpoints)
   * [Elo Rating](#elo-rating)
   * [Additional Metrics](#additional-metrics)
5. [Training & Evaluation Workflow](#training--evaluation-workflow)
6. [File Structure](#file-structure)
7. [Future Work](#future-work)

---

## Overview

This project implements a reinforcement learning agent to play Coup, a bluffing card game with hidden information and multi-agent dynamics. The goal is to train a PPO+LSTM-based policy in a Gym-style environment, using self-play and comprehensive evaluation metrics.

## Game Rules Summary

### Setup

* **Players:** 2–6 players.
* **Deck:** 15 cards total, 5 roles (Duke, Assassin, Captain, Ambassador, Contessa), 3 of each.
* **Starting Influence:** Each player begins with 2 cards (face down) and 2 coins.
* **Elimination:** Lose both Influence cards --> eliminated.

### Actions

| Action      | Cost | Role Claim | Effect                                                             |
| ----------- | ---- | ---------- | ------------------------------------------------------------------ |
| Income      | 0    | —          | +1 coin                                                            |
| Foreign Aid | 0    | —          | +2 coins (can be blocked by Duke)                                  |
| Coup        | 7    | —          | Pay 7 coins, force opponent to lose 1 card (unblockable)           |
| Tax         | 0    | Duke       | +3 coins                                                           |
| Assassinate | 3    | Assassin   | Pay 3 coins, force opponent to lose 1 card (blockable by Contessa) |
| Steal       | 0    | Captain    | +2 coins from target (blockable by Captain or Ambassador)          |
| Exchange    | 0    | Ambassador | Draw 2 cards from Court Deck, exchange any number with your hand   |

### Bluffing Mechanics

* Players **may claim** any role to perform its action, regardless of their actual cards.
* If an opponent **challenges** your claim and you can’t reveal the claimed card, you lose 1 Influence (reveal it permanently, lose the card).
* If you reveal and prove your claim, challenger loses 1 Influence and you shuffle the revealed card back into the deck and draw a replacement.

### Challenges vs Blocks

* **Challenge (calling BS on an action):**

  * If right --> actor loses 1 Influence; action is canceled.
  * If wrong --> challenger loses 1 Influence; action resolves.
* **Block (claiming to stop an action):**

  * If unchallenged --> action simply fails; no Influence lost.
  * If challenged and blocker wrong --> blocker loses 1 Influence; action still resolves.
  * If challenged and blocker right --> challenger loses 1 Influence; block holds.

#### Example Scenarios

1. **Fake block + challenge:**

   * Player 2 blocks Assassin (no Contessa), then is challenged --> loses 1 for block bluff, then assassination goes through --> total 2 Influence lost.
2. **Challenge action first:**

   * Player 2 challenges Assassin claim --> player 1 loses 1; assassination does not occur.

### Steal Blocks

* When blocking a Steal, the blocker **must specify** the card: “Block, Captain” or “Block, Ambassador.”  This reduces ambiguity and informs bluff risks.

---

## Project Architecture

### Environment Design

* **Gym-style API**: `reset()`, `step(action)`, `render()`, `observation_space`, `action_space`.
* **Turn-based multi-agent** with First-Come, First-Serve for simultaneous block/challenge reactions.
* **Partial Observability**: Agents do not see opponents’ cards; handled via recurrent policies.

### State & Observation Space

* **Private Features:** player hand (hidden)
* **Public Features:** each player’s coin count, eliminated cards, action/response history buffer.

### Action Space

* **Core Actions:** Income, Foreign Aid, Coup, Tax, Assassinate, Steal, Exchange.
* **Response Actions:**

  * Block–Foreign Aid
  * Block–Steal (Captain / Ambassador)
  * Block–Assassinate
  * Challenge–Action (specify which)
  * Challenge–Block (specify which)
  * Pass

### Reward Shaping

* **Terminal Reward:** +1 for game win, –1 for loss.
* **Shaping Bonuses/Penalties:**

  * * for successful bluff challenge.
  * – for unsuccessful challenge.
  * * small for gaining coins; – small for losing coins.
  * * for eliminating opponent’s Influence.
  * – for losing your Influence.
  * * entropy bonus early training (encourage exploration).
  * – small per-turn penalty (discourage stalling).

### Algorithm Choice

* **PPO + LSTM:** on-policy actor-critic with recurrent hidden state for partial observability. I considered DQN but from what I can tell, it struggles with non-observability (not being able to know the opponent state)
* **Why:** Stable updates, handles non-stationarity, supports memory of history.

### Compute Considerations

* **Prototype on CPU** (MacBook/iMac, what I'm currently on); expect slower training.
* **Scale to GPU** via SSH to Linux/GPU server once prototype validated.

---

## Evaluation Metrics

### Baselines & Checkpoints

* **Random Policy:** measure win-rate vs purely random moves.
* **Checkpoint Policies:** archive snapshots (e.g., 10k, 50k, 100k games) to evaluate against past selves.

### Elo Rating

* Compute Elo for head-to-head matches:
  $E_A = 1 / (1 + 10^{(R_B - R_A)/400})$
  $R_A' = R_A + K (S - E_A)$
* **K-factor:** e.g., 32.
* Update ratings over matches vs random, checkpoints, current.

### Additional Metrics

* **Average Game Length** (turns).
* **Challenge/Block Accuracy** (true positive vs false positive rates).
* **Resource Efficiency** (coins spent per elimination).

---

## Training & Evaluation Workflow

1. **Environment & Agent Setup**: implement `CoupEnv` and `PPO+LSTM` agent.
2. **Self-Play Training**: run multi-agent PPO, logging metrics and game replays.
3. **Checkpointing**: save model weights at scheduled intervals.
4. **Evaluation**: run `evaluate.py` to compute win-rates, Elo, and other metrics against frozen opponent pool.
5. **Iterate**: refine reward signals, features, hyperparameters.

---

## File Structure

```
coup_rl/
├── configs/             # YAML/JSON config files
├── envs/                # Gym environment (coup_env.py)
├── agents/              # Network & PPO-LSTM implementation
├── train/               # Training scripts (train_agent.py)
├── eval/                # Evaluation & Elo (evaluate.py, elo.py)
├── utils/               # Logging, metrics, replay buffer
├── scripts/             # Shell scripts (run_training.sh)
├── logs/                # Logs & game replays
├── notebooks/           # Analysis & visualization
├── requirements.txt     # Dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

---

## Future Work

* **Human vs Agent Frontend:** web or desktop UI after agent competency.
* **Curriculum Learning:** scale from 2-player to 6-player games.
* **Belief-State Models:** integrate explicit probability tracking of opponent cards.
* **Advanced Multi-Agent Techniques:** league training, opponent modeling.

---

