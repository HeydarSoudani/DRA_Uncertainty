"""Belief model training with the agent frozen (Agent-BRACE, arXiv 2605.11436,
adapted to retrieval): the trained model is the belief model of
``belief_model_component``, one model with two calls: init (the criteria
extraction, f(query) -> C) and update (the belief updater,
b_t = f(query, b_{t-1}, o_t)).  The query targeting scorer stays frozen.

Entry point: ``experiments/training/belief_model_training.py --phase N``.

Module layout:
    config        settings per phase, Split, RunPaths
    episodes      episodes from monitor runs, BeliefTracker, teacher replay
    trajectories  phase 1: teacher trajectories and SFT samples (training pool)
    sft           phase 2: SFT warm-up (LoRA, merged)
    rollout       vLLM generation with log-probs, sleep / weight reload
    scoring       measures of a belief model against the gold (teacher scores, validation)
    env           phase 3 environment: teacher-forced updater steps, GRPO groups, batch sampler
    rewards       phase 3 reward of one belief update
    ppo           PPO losses (from veRL)
    trainer       PPO update (from Agent-BRACE's BeliefStateLMTrainer)
    rl            phase 3 loop
"""
