"""DRA agent training infrastructure (SFT + async GRPO).

This package holds the *training* pipeline plumbing.  It is a sibling of the
finalized *inference* pipeline (``experiments/dra_inference.py`` +
``deep_research_agents`` / ``searcher_component`` / ``reasoner_component`` /
``uncertainty_estimator``) and consumes those components **read-only** — nothing
here mutates the inference code path.

Design stance:
  * INFRASTRUCTURE FIRST.  The rollout orchestration, trajectory buffer, loop
    drivers, and component interfaces are implemented here.
  * AGENT-AGNOSTIC.  An agent plugs in through ``rollout.env.AgentEnv`` (its
    prompts, turn parsing, observations, end conditions); the environment
    lives with the agent's entry script.
  * PER-TURN SAMPLES.  Every policy call is one training sample (context
    masked, generation trained) carrying the trajectory's GRPO advantage.
  * Still TODO behind stable interfaces: the GRPO update and weight sync
    (veRL), the SFT backend (veRL SFT), and process rewards (the reward is
    outcome only for now).
  * A ``mock`` policy / tool / trainer let the ENTIRE loop run on CPU with no
    GPUs, so masking / group formation / advantage shaping can be validated
    before any real training compute is wired.

Entry point: ``training.pipeline.run``, called by an agent's training script
(none at the moment).
"""

__all__ = ["config", "pipeline"]
