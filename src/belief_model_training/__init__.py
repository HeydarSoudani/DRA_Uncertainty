"""Belief model training (agent frozen), following Agent-BRACE (arXiv 2605.11436).

One student model (Qwen3.5-4B) learns the belief model's calls with the
inference prompts: belief update, final score, and (own-init RL) belief
init.  Five phases, ``experiments/training/train_belief_model.py --phase N``:

    1 data          build a training dataset (data/build.py)
    2 trajectories  dra_inference: generate on a split, the teacher's
                    beliefs on sft and val (trajectories/)
    3 sft           the teacher's gated replies, LoRA SFT, merged (sft/)
    4 rl            GRPO over belief chains, reward = Brier of the final
                    score behind a format gate (rl/, reward.py)
    5 evaluate      dra_inference --stage belief with a belief model (evaluate/)

Shared across phases:
    config      settings (experiments/configs/belief_training.yaml), run paths
    runs        the episodes of the phase 2 runs, their documents
    chain       one chain = the inference BeliefModel with a recording client
    reward      gate and Brier reward
    student     chat prompt, weights, tokenization
    inference   dra_inference child process

The belief model, the agents and dra_inference are used as they are: a
chain swaps only the LLM client of ``belief_model_component.BeliefModel``.
"""
