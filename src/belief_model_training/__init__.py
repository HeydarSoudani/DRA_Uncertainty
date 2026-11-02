"""Belief model training with the agent frozen (Agent-BRACE, arXiv 2605.11436,
adapted to retrieval): the trained model is the belief model of
``belief_model_component``, one model with two calls: init (the criteria
extraction, f(query) -> C) and update (the belief updater,
b_t = f(query, b_{t-1}, o_t)).  The query targeting scorer stays frozen.

Entry point: ``experiments/training/belief_model_training.py --phase N``.

Module layout:
    config        settings per phase, Split, RunPaths
    episodes      episodes from trajectories (+ the teacher's belief files), BeliefTracker, teacher replay
    runs          the phase 1 runs of a pool split, read back as episodes (phases 2, 3)
    trajectories  phase 1: dra_inference on a pool split (sft, val: monitor; rl: off)
    sft           phase 2: SFT samples and teacher scores, SFT warm-up (LoRA, merged)
    rollout       vLLM generation with log-probs, sleep / weight reload
    candidate_gold the gold of the reasoner side: answer candidates and agent turns against the gold answers
    scoring       measures of a belief model against the gold (teacher scores, validation)
    env           phase 3 environment: the student's belief chains over whole episodes
                  (GRPO groups), episode sampler
    rewards       phase 3 reward of one belief chain
    ppo           PPO losses (from veRL)
    trainer       PPO update (from Agent-BRACE's BeliefStateLMTrainer)
    rl            phase 3 loop
"""
