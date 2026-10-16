"""Belief model training with the agent frozen (Agent-BRACE, arXiv 2605.11436,
adapted to retrieval): the trained call is the belief updater
b_t = f(query, b_{t-1}, o_t) of ``belief_model_component``.

Entry point: ``experiments/belief_model_training.py --phase N``.

Module layout:
    config        settings per phase, Split, RunPaths
    queries       phase 0: training queries (TODO)
    episodes      episodes from monitor runs, BeliefTracker, teacher replay
    trajectories  phase 1: teacher trajectories and SFT samples
    sft           phase 2: SFT warm-up (LoRA, merged)
    rollout       vLLM generation with log-probs, sleep / weight reload
    env           phase 3 environments (replay; online TODO)
    rewards       phase 3 rewards (TODO)
    ppo           PPO losses (from veRL)
    trainer       PPO update (from Agent-BRACE's BeliefStateLMTrainer)
    rl            phase 3 loop
    evaluate      phase 4: off / teacher / student with the existing evaluators
"""
