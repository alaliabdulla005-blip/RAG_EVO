"""Fitness function for the RAG-Evo genetic algorithm.

Scalar-weighted fitness:
    F(x) = w_q·Q + w_g·G + w_r·R − w_c·C − w_l·L

Positive weights reward quality; negative weights penalise cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_evo.evaluation.harness import AggregateMetrics


@dataclass(frozen=True)
class FitnessWeights:
    """Weights for the scalar fitness function.

    Args:
        w_q: Answer quality weight (mean F1).
        w_g: Grounding quality weight (mean citation F1).
        w_r: Retrieval quality weight (mean retrieval recall@k).
        w_c: Cost penalty weight (input tokens / budget).
        w_l: Latency penalty weight (mean latency / budget).
        token_budget: Total input token budget for cost normalisation.
        latency_budget_ms: Latency budget for latency normalisation.
    """

    w_q: float = 0.4
    w_g: float = 0.3
    w_r: float = 0.2
    w_c: float = 0.05
    w_l: float = 0.05
    token_budget: int = 32_500
    latency_budget_ms: float = 5000.0


@dataclass(frozen=True)
class FitnessComponents:
    """Individual normalised components of the fitness computation.

    Args:
        q_norm: Normalised answer quality (mean F1).
        g_norm: Normalised grounding quality (mean citation F1).
        r_norm: Normalised retrieval quality (mean recall@k).
        c_norm: Normalised cost (total input tokens / budget).
        l_norm: Normalised latency (mean latency / budget).
        fitness: Final scalar fitness value.
    """

    q_norm: float
    g_norm: float
    r_norm: float
    c_norm: float
    l_norm: float
    fitness: float


def compute_fitness_components(
    aggregate: AggregateMetrics,
    weights: FitnessWeights,
) -> FitnessComponents:
    """Compute all fitness components and the final scalar fitness.

    Args:
        aggregate: Aggregate metrics from one pipeline evaluation.
        weights: Fitness function weights (includes budgets).

    Returns:
        FitnessComponents with all normalised values and fitness.
    """
    q_norm = aggregate.mean_f1
    g_norm = aggregate.mean_citation_f1
    r_norm = aggregate.mean_retrieval_recall_at_k
    c_norm = min(
        aggregate.total_input_tokens / weights.token_budget, 1.0
    )
    l_norm = min(
        aggregate.mean_latency_ms / weights.latency_budget_ms, 1.0
    )

    fitness = (
        weights.w_q * q_norm
        + weights.w_g * g_norm
        + weights.w_r * r_norm
        - weights.w_c * c_norm
        - weights.w_l * l_norm
    )

    return FitnessComponents(
        q_norm=q_norm,
        g_norm=g_norm,
        r_norm=r_norm,
        c_norm=c_norm,
        l_norm=l_norm,
        fitness=fitness,
    )


def compute_fitness(
    aggregate: AggregateMetrics,
    weights: FitnessWeights,
) -> float:
    """Compute the scalar fitness value.

    Convenience wrapper around compute_fitness_components.

    Args:
        aggregate: Aggregate metrics from one pipeline evaluation.
        weights: Fitness function weights (includes budgets).

    Returns:
        Scalar fitness value.
    """
    return compute_fitness_components(aggregate, weights).fitness
