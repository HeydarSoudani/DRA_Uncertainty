"""What every training stage uses.

    config      settings (experiments/configs/belief_training.yaml), run paths
    runs        the episodes of the generated runs, their documents
    chain       one chain = the inference BeliefModel with a recording client
    reward      gate and Brier reward
    student     chat prompt, weights, tokenization
    inference   dra_inference child process

Nothing here knows a stage: the stages import from here, never the reverse.
(The config's ``shared`` section is the settings counterpart: the settings
every stage reads.)
"""
