"""Genetic algorithm main loop for RAG-Evo.

Evolves pipeline configurations by evaluating them through
ExperimentRunner and selecting for high fitness.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        max_workers: Number of parallel evaluation threads per generation.
            1 = sequential (default). >1 enables ThreadPoolExecutor.
    """

    population_size: int = 10
    generations: int = 15
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_count: int = 1
    seed: int = 42
    max_workers: int = 1


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


def load_latest_checkpoint(
    output_dir: Path,
    search_space: SearchSpace,
) -> (
    tuple[
        list[GenerationRecord],
        list[MicroPilotConfig | FullV1Config],
        list[float],
        tuple,
    ]
    | None
):
    """Load the most recent per-generation checkpoint from disk.

    Reads all gen_NNN/ subdirectories, reconstructs GenerationRecord
    objects, the last generation's population and fitnesses, and the
    saved RNG state so the GA can resume deterministically.

    Args:
        output_dir: Directory that contains gen_NNN/ subdirectories.
        search_space: SearchSpace used for this GA run. Its version
            determines whether to reconstruct MicroPilotConfig or
            FullV1Config objects.

    Returns:
        (records, population, fitnesses, rng_state) if at least one
        checkpoint generation exists, else None.
    """
    gen_dirs = sorted(output_dir.glob("gen_*"))
    if not gen_dirs:
        return None

    records: list[GenerationRecord] = []
    for gen_dir in gen_dirs:
        record_path = gen_dir / "record.json"
        if record_path.exists():
            d = json.loads(record_path.read_text(encoding="utf-8"))
            records.append(GenerationRecord(**d))

    if not records:
        return None

    last_gen_dir = gen_dirs[-1]

    pop_path = last_gen_dir / "population.json"
    pop_data = json.loads(pop_path.read_text(encoding="utf-8"))
    if search_space._version == "micro_pilot":
        population: list[MicroPilotConfig | FullV1Config] = [
            MicroPilotConfig.from_dict(entry["config"])
            for entry in pop_data
        ]
    else:
        population = [
            FullV1Config.from_dict(entry["config"])
            for entry in pop_data
        ]
    fitnesses = [entry["fitness"] for entry in pop_data]

    rng_path = last_gen_dir / "rng_state.json"
    raw = json.loads(rng_path.read_text(encoding="utf-8"))
    rng_state: tuple = (raw[0], tuple(raw[1]), raw[2])

    logger.info(
        "Loaded checkpoint: %d generation(s) complete, last=%s",
        len(records),
        last_gen_dir.name,
    )
    return records, population, fitnesses, rng_state


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
        self._max_workers = ga_config.max_workers

        # Build a pool of ExperimentRunner instances — one per worker
        # thread — so each thread has its own index cache and avoids
        # TOCTOU races on the shared dict. All runners share the same
        # ResultCache (disk-backed, lock-protected) and GeneratorWrapper
        # (httpx is thread-safe). We access private attributes here
        # intentionally, as this is within the same package.
        self._runner_pool: queue.SimpleQueue | None = None
        if self._max_workers > 1:
            self._runner_pool = queue.SimpleQueue()
            for _ in range(self._max_workers):
                self._runner_pool.put(
                    ExperimentRunner(
                        wrapper=runner._wrapper,
                        harness=runner._harness,
                        cache=runner._cache,
                        reranker=runner._reranker,
                    )
                )

    def _evaluate(
        self, config: MicroPilotConfig | FullV1Config,
    ) -> tuple[float, FitnessComponents]:
        """Evaluate one config through the full pipeline (sequential path)."""
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

    def _evaluate_one(
        self,
        idx: int,
        config: MicroPilotConfig | FullV1Config,
        gen_idx: int,
        gc: GAConfig,
    ) -> tuple[int, float, FitnessComponents]:
        """Evaluate one individual in a worker thread (parallel path).

        Borrows a dedicated ExperimentRunner from the pool so each thread
        has an independent index cache. Returns the individual's position
        index so results can be reassembled in population order.

        Exceptions are caught here — never propagated — so the
        ThreadPoolExecutor never sees a failed future.

        Args:
            idx: Position of this individual in the population list.
            config: Pipeline configuration to evaluate.
            gen_idx: Current zero-based generation index (for logging).
            gc: GAConfig (for logging totals).

        Returns:
            (idx, fitness, components). On exception, fitness=-1.0 and
            components are zeroed.
        """
        # Shorten the default thread name for readable logs:
        # "ThreadPoolExecutor-0_2" → "T2"
        tname = threading.current_thread().name
        tag = "T" + tname.rsplit("_", 1)[-1]

        logger.info(
            "[%s] Gen %d/%d, individual %d/%d",
            tag,
            gen_idx + 1,
            gc.generations,
            idx + 1,
            gc.population_size,
        )

        assert self._runner_pool is not None
        runner = self._runner_pool.get()
        try:
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
            result = runner.run(run_config)
            components = compute_fitness_components(
                result.aggregate_metrics, self._weights
            )
            return idx, components.fitness, components
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Individual %d evaluation failed: %s",
                tag,
                idx + 1,
                exc,
            )
            zero = FitnessComponents(
                q_norm=0.0,
                g_norm=0.0,
                r_norm=0.0,
                c_norm=0.0,
                l_norm=0.0,
                fitness=-1.0,
            )
            return idx, -1.0, zero
        finally:
            self._runner_pool.put(runner)

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
        """Run the full GA loop, resuming from checkpoint if available.

        If output_dir is set and contains gen_NNN/ checkpoint directories,
        the run automatically resumes from the latest saved state so that
        long experiments survive interruptions without losing work.

        Returns:
            List of GenerationRecord, one per generation.
        """
        if self._output_dir is not None:
            checkpoint = load_latest_checkpoint(
                self._output_dir, self._space
            )
            if checkpoint is not None:
                records, population, fitnesses, rng_state = checkpoint
                logger.info(
                    "Resuming from gen %d checkpoint",
                    len(records) - 1,
                )
                self._rng.setstate(rng_state)
                return self.resume(records, population, fitnesses)
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

            if self._max_workers > 1:
                # Parallel evaluation: submit all individuals concurrently,
                # collect results in original population order.
                results_map: dict[int, tuple[float, FitnessComponents]] = {}
                with ThreadPoolExecutor(
                    max_workers=self._max_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            self._evaluate_one, i, config, gen_idx, gc
                        )
                        for i, config in enumerate(population)
                    ]
                    for future in as_completed(futures):
                        i_done, fit, comp = future.result()
                        results_map[i_done] = (fit, comp)
                fitnesses = [
                    results_map[i][0] for i in range(len(population))
                ]
                components_list = [
                    results_map[i][1] for i in range(len(population))
                ]
            else:
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

        # Save RNG state so run() can resume deterministically after
        # interruption. json.dumps serialises tuples as JSON arrays;
        # load_latest_checkpoint converts them back to tuples on load.
        (gen_dir / "rng_state.json").write_text(
            json.dumps(self._rng.getstate()),
            encoding="utf-8",
        )
