"""Genetic algorithm main loop for RAG-Evo.

Evolves pipeline configurations by evaluating them through
ExperimentRunner and selecting for high fitness.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_evo.config.schema import FullV1Config, MicroPilotConfig
from rag_evo.config.space import SearchSpace
from rag_evo.ga.fitness import (
    FitnessComponents,
    FitnessWeights,
    compute_fitness_components,
)
from rag_evo.ga.operators import mutate, tournament_select, uniform_crossover
from rag_evo.runner.experiment_runner import (
    ExperimentRunner,
    RunConfig,
    generate_run_id,
)

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm.

    Args:
        population_size: Number of individuals per generation.
        generations: Total number of generations to evolve.
        tournament_size: Individuals sampled per tournament.
        crossover_rate: Probability of applying crossover.
        mutation_rate: Per-gene mutation probability.
        elite_count: Number of best individuals carried forward.
        seed: Random seed for reproducibility.
    """

    population_size: int = 10
    generations: int = 15
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_count: int = 1
    seed: int = 42


@dataclass(frozen=True)
class GenerationRecord:
    """Records statistics from one GA generation.

    Args:
        generation_idx: Zero-based generation index.
        best_fitness: Highest fitness in this generation.
        mean_fitness: Mean fitness across the population.
        worst_fitness: Lowest fitness in this generation.
        best_config: Dict representation of the best config.
        best_components: Dict of FitnessComponents for the best.
        elapsed_s: Wall-clock time for this generation in seconds.
    """

    generation_idx: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_config: dict
    best_components: dict
    elapsed_s: float


class GeneticAlgorithm:
    """GA loop that evolves pipeline configs via ExperimentRunner.

    Args:
        ga_config: GA hyperparameters.
        run_config_template: RunConfig used as a template.
            pipeline_config and run_id are overwritten per individual.
        runner: ExperimentRunner instance (shared).
        search_space: SearchSpace defining gene bounds.
        fitness_weights: Weights for the fitness function
            (includes token_budget and latency_budget_ms).
        output_dir: Directory to save per-generation artifacts.
    """

    def __init__(
        self,
        ga_config: GAConfig,
        run_config_template: RunConfig,
        runner: ExperimentRunner,
        search_space: SearchSpace,
        fitness_weights: FitnessWeights,
        output_dir: Path | None = None,
    ) -> None:
        self._ga_config = ga_config
        self._template = run_config_template
        self._runner = runner
        self._space = search_space
        self._weights = fitness_weights
        self._output_dir = output_dir
        self._rng = random.Random(ga_config.seed)

    def _evaluate(
        self, config: MicroPilotConfig | FullV1Config,
    ) -> tuple[float, FitnessComponents]:
        """Evaluate one config through the full pipeline."""
        run_id = generate_run_id(config)
        run_config = RunConfig(
            pipeline_config=config,
            run_id=run_id,
            qa_records=self._template.qa_records,
            corpus_documents=self._template.corpus_documents,
            cache_dir=self._template.cache_dir,
            results_dir=self._template.results_dir,
            model_id=self._template.model_id,
            seed=self._template.seed,
        )
        result = self._runner.run(run_config)
        components = compute_fitness_components(
            result.aggregate_metrics,
            self._weights,
        )
        return components.fitness, components

    def _init_population(
        self,
    ) -> list[MicroPilotConfig | FullV1Config]:
        """Create initial random population."""
        return [
            self._space.sample_random(self._rng)
            for _ in range(self._ga_config.population_size)
        ]

    def _next_generation(
        self,
        population: list[MicroPilotConfig | FullV1Config],
        fitnesses: list[float],
    ) -> list[MicroPilotConfig | FullV1Config]:
        """Create next generation via selection, crossover, mutation."""
        gc = self._ga_config

        # Elitism: carry forward best individuals unchanged
        ranked = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )
        new_pop: list[MicroPilotConfig | FullV1Config] = [
            population[ranked[i]] for i in range(gc.elite_count)
        ]

        while len(new_pop) < gc.population_size:
            p1 = tournament_select(
                population, fitnesses, gc.tournament_size, self._rng
            )
            p2 = tournament_select(
                population, fitnesses, gc.tournament_size, self._rng
            )
            if self._rng.random() < gc.crossover_rate:
                child = uniform_crossover(
                    p1, p2, self._rng, self._space
                )
            else:
                child = p1
            child = mutate(
                child, self._rng, gc.mutation_rate, self._space
            )
            new_pop.append(child)

        return new_pop[: gc.population_size]

    def run(self) -> list[GenerationRecord]:
        """Run the full GA loop from scratch.

        Returns:
            List of GenerationRecord, one per generation.
        """
        population = self._init_population()
        return self._evolve(population, start_gen=0)

    def resume(
        self,
        records: list[GenerationRecord],
        population: list[MicroPilotConfig | FullV1Config],
        fitnesses: list[float],
    ) -> list[GenerationRecord]:
        """Resume GA from a saved state.

        Args:
            records: GenerationRecords from prior generations.
            population: Last generation's population.
            fitnesses: Last generation's fitness values.

        Returns:
            Extended list of GenerationRecords including new ones.
        """
        start_gen = len(records)
        next_pop = self._next_generation(population, fitnesses)
        return self._evolve(
            next_pop,
            start_gen=start_gen,
            prior_records=records,
        )

    def _evolve(
        self,
        population: list[MicroPilotConfig | FullV1Config],
        start_gen: int = 0,
        prior_records: list[GenerationRecord] | None = None,
    ) -> list[GenerationRecord]:
        """Core evolution loop.

        Args:
            population: Initial population for this run.
            start_gen: Generation index to start from.
            prior_records: Records from prior generations (for resume).

        Returns:
            Complete list of GenerationRecords.
        """
        records = list(prior_records or [])
        gc = self._ga_config

        for gen_idx in range(start_gen, gc.generations):
            t0 = time.perf_counter()
            fitnesses: list[float] = []
            components_list: list[FitnessComponents] = []

            for i, config in enumerate(population):
                logger.info(
                    "Gen %d/%d, individual %d/%d",
                    gen_idx + 1,
                    gc.generations,
                    i + 1,
                    gc.population_size,
                )
                fitness, comp = self._evaluate(config)
                fitnesses.append(fitness)
                components_list.append(comp)

            best_idx = max(
                range(len(fitnesses)),
                key=lambda j: fitnesses[j],
            )
            elapsed = time.perf_counter() - t0

            record = GenerationRecord(
                generation_idx=gen_idx,
                best_fitness=max(fitnesses),
                mean_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=min(fitnesses),
                best_config=asdict(population[best_idx]),
                best_components=asdict(components_list[best_idx]),
                elapsed_s=elapsed,
            )
            records.append(record)

            logger.info(
                "Gen %d: best=%.4f mean=%.4f worst=%.4f (%.1fs)",
                gen_idx,
                record.best_fitness,
                record.mean_fitness,
                record.worst_fitness,
                elapsed,
            )

            if self._output_dir is not None:
                self._save_generation(
                    record, population, fitnesses
                )

            # Create next generation (unless last)
            if gen_idx < gc.generations - 1:
                population = self._next_generation(
                    population, fitnesses
                )

        return records

    def _save_generation(
        self,
        record: GenerationRecord,
        population: list[MicroPilotConfig | FullV1Config],
        fitnesses: list[float],
    ) -> None:
        """Save generation artifacts to disk."""
        assert self._output_dir is not None
        gen_dir = (
            self._output_dir / f"gen_{record.generation_idx:03d}"
        )
        gen_dir.mkdir(parents=True, exist_ok=True)

        (gen_dir / "record.json").write_text(
            json.dumps(asdict(record), indent=2) + "\n",
            encoding="utf-8",
        )

        pop_data = [
            {"config": asdict(c), "fitness": f}
            for c, f in zip(population, fitnesses)
        ]
        (gen_dir / "population.json").write_text(
            json.dumps(pop_data, indent=2) + "\n",
            encoding="utf-8",
        )
