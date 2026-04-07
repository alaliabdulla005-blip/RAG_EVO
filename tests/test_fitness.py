"""Tests for src/rag_evo/ga/fitness.py."""

from rag_evo.evaluation.harness import AggregateMetrics
from rag_evo.ga.fitness import (
    FitnessComponents,
    FitnessWeights,
    compute_fitness,
    compute_fitness_components,
)


def _agg(
    mean_f1: float = 0.5,
    mean_citation_f1: float = 0.3,
    mean_retrieval_recall_at_k: float = 0.6,
    total_input_tokens: int = 1000,
    total_output_tokens: int = 1000,
    mean_latency_ms: float = 200.0,
) -> AggregateMetrics:
    return AggregateMetrics(
        num_records=10,
        mean_em=0.4,
        mean_f1=mean_f1,
        mean_citation_precision=0.3,
        mean_citation_recall=0.3,
        mean_citation_f1=mean_citation_f1,
        mean_retrieval_recall_at_k=mean_retrieval_recall_at_k,
        mean_input_tokens=100.0,
        mean_output_tokens=10.0,
        mean_latency_ms=mean_latency_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def test_fitness_weights_defaults() -> None:
    w = FitnessWeights()
    assert w.w_q == 0.4
    assert w.w_g == 0.3
    assert w.w_r == 0.2
    assert w.w_c == 0.05
    assert w.w_l == 0.05
    assert w.token_budget == 32_500
    assert w.latency_budget_ms == 5000.0


def test_compute_fitness_basic() -> None:
    agg = _agg()
    w = FitnessWeights(token_budget=10_000, latency_budget_ms=1000.0)
    f = compute_fitness(agg, w)
    # Q=0.5*0.4=0.20, G=0.3*0.3=0.09, R=0.6*0.2=0.12
    # C=1000/10000=0.1 -> 0.1*0.05=0.005
    # L=200/1000=0.2 -> 0.2*0.05=0.01
    # F = 0.20 + 0.09 + 0.12 - 0.005 - 0.01 = 0.395
    assert abs(f - 0.395) < 1e-9


def test_compute_fitness_components_returns_dataclass() -> None:
    agg = _agg()
    w = FitnessWeights(token_budget=10_000, latency_budget_ms=1000.0)
    comp = compute_fitness_components(agg, w)
    assert isinstance(comp, FitnessComponents)
    assert comp.q_norm == 0.5
    assert comp.g_norm == 0.3
    assert comp.r_norm == 0.6
    assert abs(comp.c_norm - 0.1) < 1e-9
    assert abs(comp.l_norm - 0.2) < 1e-9
    assert abs(comp.fitness - 0.395) < 1e-9


def test_cost_norm_clamped_to_one() -> None:
    """total_input_tokens > budget should clamp c_norm to 1.0."""
    agg = _agg(total_input_tokens=999_999)
    w = FitnessWeights(token_budget=100)
    comp = compute_fitness_components(agg, w)
    assert comp.c_norm == 1.0


def test_latency_norm_clamped_to_one() -> None:
    """mean_latency_ms > budget should clamp l_norm to 1.0."""
    agg = _agg(mean_latency_ms=99_999.0)
    w = FitnessWeights(latency_budget_ms=100.0)
    comp = compute_fitness_components(agg, w)
    assert comp.l_norm == 1.0


def test_perfect_quality_zero_cost() -> None:
    """All quality at 1.0, zero cost/latency -> maximum fitness."""
    agg = _agg(
        mean_f1=1.0,
        mean_citation_f1=1.0,
        mean_retrieval_recall_at_k=1.0,
        total_input_tokens=0,
        mean_latency_ms=0.0,
    )
    w = FitnessWeights()
    f = compute_fitness(agg, w)
    # 0.4 + 0.3 + 0.2 - 0 - 0 = 0.9
    assert abs(f - 0.9) < 1e-9


def test_custom_weights() -> None:
    agg = _agg(mean_f1=1.0, mean_citation_f1=0.0,
               mean_retrieval_recall_at_k=0.0,
               total_input_tokens=0, mean_latency_ms=0.0)
    w = FitnessWeights(w_q=1.0, w_g=0.0, w_r=0.0, w_c=0.0, w_l=0.0)
    f = compute_fitness(agg, w)
    assert abs(f - 1.0) < 1e-9
