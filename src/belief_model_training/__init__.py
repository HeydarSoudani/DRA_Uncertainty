"""Belief model training (agent frozen), following Agent-BRACE (arXiv 2605.11436).

One student model (``shared.student``, Qwen3.5-4B) learns the belief model's
calls with the inference prompts: belief update, final score, and (own-init
RL) belief init.  Eight stages,
``experiments/training/train_belief_model.py --stage NAME``:

    data_prepare    the questions of a training dataset worth its gold
                    annotation (indexing_corpus_dataset.training_datasets)
    data_relevance  its gold annotation, qrels and sft / rl / val splits
                    (indexing_corpus_dataset.training_datasets)
    generate        dra_inference --stage generate on a split (trajectories/)
    teacher         dra_inference --stage belief with the teacher on sft and
                    val (teacher/)
    filter          what is trained on: SFT samples from the teacher's gated
                    replies, RL episodes (filtering/)
    sft             LoRA SFT on the filtered samples, merged (sft/)
    rl              GRPO over belief chains of the filtered episodes, reward =
                    Brier of the final score behind a format gate (rl/,
                    shared/reward.py)
    evaluate        dra_inference --stage belief with a belief model (evaluate/)

Shared across stages (shared/):
    config      settings (experiments/configs/belief_training.yaml), run paths
    runs        the episodes of the generated runs, their documents
    chain       one chain = the inference BeliefModel with a recording client
    reward      gate and Brier reward
    student     chat prompt, weights, tokenization
    inference   dra_inference child process

The belief model, the agents and dra_inference are used as they are: a
chain swaps only the LLM client of ``belief_model_component.BeliefModel``.
"""
