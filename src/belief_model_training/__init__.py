"""Belief model training with the agent frozen (Agent-BRACE, arXiv 2605.11436,
adapted to retrieval): the trained model is the belief update of
``belief_model_component``, one call per step,
b_t = f(query, b_{t-1}, agent turn t, new passages), on the pool's gold
criteria (the criteria init call is not trained).  It is rewarded over
whole trajectories, per criterion: the assignment of the sources to it
(its gold documents, the turns that state its gold answer) and the
calibration of its status at every step; plus the share of well-formed
calls and, at the end, the certainty score of the final beliefs
against the correctness of the agent's final answer.

Entry point: ``experiments/training/belief_model_training.py --phase N``.

Module layout:
    config        settings per phase, Split, RunPaths
    episodes      episodes from trajectories (+ the teacher's belief files), BeliefTracker, teacher replay
    runs          the phase 1 runs of a pool split, read back as episodes (phases 2, 3)
    trajectories  phase 1: dra_inference on a pool split (sft, val: monitor; rl: off)
    sft           phase 2: SFT samples (update calls) and teacher scores, SFT warm-up (LoRA, merged)
    rollout       vLLM generation with log-probs, sleep / weight reload
    nugget_gold   the gold of the belief: candidates and spans against the gold answers, which criteria a turn states
    scoring       per-criterion scores and measures of a belief model against the gold (teacher scores,
                  rewards, validation)
    env           phase 3 environment: the student's belief chains over whole episodes
                  (GRPO groups), episode sampler
    rewards       phase 3 reward of one belief chain
    ppo           PPO losses (from veRL)
    trainer       PPO update (from Agent-BRACE's BeliefStateLMTrainer)
    rl            phase 3 loop
"""
