"""Factories of the pipeline's components: retriever, rerankers, search tool,
agent and belief model.

Shared by the main process (one worker) and the GPU workers
(``pipeline.workers._init_worker``), so each component is built in one
place.
"""

import os
from typing import Optional, Tuple

from indexing_corpus_dataset.layout import DATASET_SPECS, query_max_length
from utils.config import (
    OPENROUTER_BASE_URL,
    RetrieverConfig,
    effective_max_output_tokens,
    get_reranker_configs,
    resolve_agent_backend,
)
from utils.text_utils import MAX_PASSAGE_TOKENS


# ===========================================================================
# Searcher: retriever, rerankers, search tool
# ===========================================================================

def setup_retriever_from_args(args):
    """Instantiate the local retriever from parsed CLI args."""
    from searcher_component.retriever import (
        BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever,
    )

    config = RetrieverConfig(
        retriever_name=args.retriever,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        topk=args.top_k,
    )
    config.retrieval_query_max_length = query_max_length(args.dataset, args.retriever)
    if args.retriever == "bm25":
        return BM25Retriever(config)
    elif args.retriever in ("spladepp", "spladev3"):
        return SPLADERetriever(config)
    elif args.retriever in ["rerank_l6", "rerank_l12"]:
        return RerankRetriever(config)
    else:
        return DenseRetriever(config)


def build_rerankers(post_retrieval_reranker: str, post_fusion_reranker: str, rerank_top_k: int) -> Tuple:
    """The post-retrieval and post-fusion rerankers of their CLI names
    (``"null"``: none, None)."""
    from searcher_component.rerankers import build_reranker_from_config

    configs = get_reranker_configs(rerank_top_k)
    post_retrieval = (build_reranker_from_config(post_retrieval_reranker, configs)
                      if post_retrieval_reranker != "null" else None)
    post_fusion = (build_reranker_from_config(post_fusion_reranker, configs)
                   if post_fusion_reranker != "null" else None)
    return post_retrieval, post_fusion


def build_search_tool(retriever, *, post_retrieval_reranker=None, post_fusion_reranker=None,
                      rerank_top_k: int, retrieval_input: str, post_fusion_reranker_input: str):
    """The agents' search tool over *retriever* (None without a retriever)."""
    if retriever is None:
        return None
    from searcher_component import RetrievalSearchTool

    return RetrievalSearchTool(
        retriever=retriever,
        post_retrieval_reranker=post_retrieval_reranker,
        post_fusion_reranker=post_fusion_reranker,
        rerank_top_k=rerank_top_k,
        retrieval_input=retrieval_input,
        post_fusion_reranker_input=post_fusion_reranker_input,
    )


# ===========================================================================
# Belief model
# ===========================================================================

def build_belief_model(
    *,
    dataset: str,
    belief_model: str,
    belief_source,
    qrels=None,
    max_passage_tokens: int = MAX_PASSAGE_TOKENS,
):
    """Build the belief model of the belief stage (``pipeline.belief_stage``).

    Args:
        dataset: Dataset name (its query shape).
        belief_model: The belief model's backbone LLM, for the belief update
            and the final score (its belief init is *belief_source*'s).  A
            ``vllm/`` model is served locally (``init.factory.belief_llm``),
            e.g. a trained belief model.
        belief_source: The belief source of the run's split
            (``belief_model_component.build_belief_source``, on the same
            *belief_model*); built once and shared by the stage's workers.
        qrels: Thresholded relevance judgements, for the new-item precision
            and the recall so far.
        max_passage_tokens: Tokens of one document the belief update
            reads (its first ones), the agents' own (the run's shared setting).
    """
    from belief_model_component import BeliefModel, belief_llm, build_belief_updater, build_final_scorer

    query_shape = DATASET_SPECS[dataset].query_shape
    judge_llm = belief_llm(belief_model)
    belief_updater = build_belief_updater(judge_llm, model=belief_model, query_shape=query_shape,
                                          max_passage_tokens=max_passage_tokens)
    print(f"Belief model: belief updater {belief_updater.name}")
    # s_T: the final scorer on the same model (a short-answer query;
    # a report keeps the certainty of its final belief).
    final_scorer = build_final_scorer(judge_llm, model=belief_model, query_shape=query_shape)
    print(f"Belief model: final scorer {final_scorer.name if final_scorer else 'none (report: s_T from the belief)'}")

    return BeliefModel(
        belief_source=belief_source,
        belief_updater=belief_updater,
        qrels=qrels or {},
        final_scorer=final_scorer,
        token_meter=getattr(judge_llm, "token_meter", None),
    )


# ===========================================================================
# Agent
# ===========================================================================

def _openrouter_route(cli_name: str) -> dict:
    """The agent constructor's OpenRouter settings when *cli_name* is in the
    OpenRouter registry (0 local GPU); empty otherwise: the agent's local
    vLLM defaults (registry AGENT_API_BASE).  Resolved on the user-facing
    CLI name: the registry distinguishes oss_20b / oss_120b, which both
    collapse to the internal agent name "oss"."""
    backend, slug = resolve_agent_backend(cli_name)
    if backend != "api":
        return {}
    return {"model_url": OPENROUTER_BASE_URL, "model_name": slug, "api_key": os.getenv("OPENROUTER_API_KEY")}


def build_agent(
    policy_model: str,
    llm_model: str,
    llm_client=None,
    retriever=None,
    *,
    policy_model_cli: Optional[str] = None,
    dataset: Optional[str] = None,
    max_iteration: int = 100,
    seen_top_k: int = 5,
    verbose: bool = False,
    search_tool=None,
    use_plan: bool = False,
    max_output_tokens_total: int,
    temperature: float,
    max_extend_steps: int = 5,
    max_retries: int = 3,
    hard_mode: bool = True,
    max_passage_tokens: int = MAX_PASSAGE_TOKENS,
):
    """Instantiate an agent and attach its search tool.

    Shared by both the main process (one worker) and spawned GPU workers
    (``pipeline.workers._init_worker``) so the per-agent construction logic
    lives in one place.

    The agent-specific keyword arguments are assembled into ``agent_kwargs``
    based on ``policy_model`` and forwarded to the agent constructor.
    """
    from deep_research_agents.agents import AGENT_MAP

    model_class = AGENT_MAP[policy_model]
    cli_name = policy_model_cli or policy_model
    # The capped value (utils.config.OUTPUT_TOKEN_CAPS); run_config.json records the same.
    max_output_tokens = effective_max_output_tokens(policy_model, max_output_tokens_total)
    agent_kwargs: dict = {}
    if use_plan and policy_model == "react":
        agent_kwargs["use_plan"] = True
    if policy_model == "glm":
        agent_kwargs["max_output_tokens"] = max_output_tokens
        # OpenRouter when the agent is in the registry, else its vLLM defaults.
        agent_kwargs.update(_openrouter_route(cli_name))
    elif policy_model == "oss":
        # Cap per-call output like GLM so the prompt/history still fits inside
        # the 131072-token vLLM window (gpt-oss native max = OpenRouter's max).
        # On the OpenRouter Responses path truncation:"auto" reclaims the full
        # window regardless of this ceiling.
        agent_kwargs["max_output_tokens"] = max_output_tokens
        # OpenRouter (Responses API) when oss_20b / oss_120b is in the
        # registry; else the local vLLM server's model name.
        agent_kwargs.update(_openrouter_route(cli_name) or {"model_name": f"openai/{llm_model}"})
    elif policy_model == "tongyi":
        agent_kwargs["max_tokens_per_step"] = max_output_tokens
    elif policy_model == "cpm_explore":
        agent_kwargs["max_output_tokens"] = max_output_tokens
        # The run's resolved temperature (utils.config.resolve_temperature),
        # not a falsy-guarded default: 0.0 is a temperature a user can ask for.
        agent_kwargs["temperature"] = temperature
    elif policy_model == "cpm_report":
        agent_kwargs["max_extend_steps"] = max_extend_steps
        agent_kwargs["max_retries"] = max_retries
        agent_kwargs["hard_mode"] = hard_mode
        agent_kwargs["model_name"] = llm_model
    # Report datasets get the report prompts (task from the dataset); a
    # report has no length limit.
    if policy_model in ("oss", "glm") and dataset is not None:
        agent_kwargs["task"] = DATASET_SPECS[dataset].task

    agent = model_class(
        llm_client=llm_client,
        retriever=retriever,
        max_iteration=max_iteration,
        seen_top_k=seen_top_k,
        verbose=verbose,
        **agent_kwargs,
    )
    # One document length for every agent (and the belief model):
    # ``max_passage_tokens``, the run's shared setting.
    agent.max_passage_tokens = max_passage_tokens

    # Inject search tool into agents that inherit from BasicAgent
    # (avoids modifying every subclass constructor).
    if search_tool is not None and hasattr(agent, "search_tool"):
        agent.search_tool = search_tool

    return agent
