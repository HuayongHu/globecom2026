from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np

try:
    from scipy.optimize import differential_evolution
except Exception:
    differential_evolution = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
except Exception:
    RandomForestRegressor = None
    MultiOutputRegressor = None

from .models import Environment, Waveform, WaveformConstraints
from .robust import RobustObjective, RobustResult
from .rag import WaveformLibrary, RuleBasedDesigner
from .llm_client import LLMWaveformClient


@dataclass
class DesignResult:
    method: str
    waveform: Waveform
    robust: RobustResult
    runtime_sec: float
    eval_count: int
    history: List[Dict[str, float]]
    extra: Dict[str, float | int | str]

    def row(self, scenario_id: str, env: Environment) -> dict:
        return {
            "scenario_id": scenario_id,
            "method": self.method,
            "runtime_sec": self.runtime_sec,
            "eval_count": self.eval_count,
            **env.to_dict(),
            **self.waveform.to_dict(),
            **self.robust.to_dict(),
            **self.extra,
        }


class CrossEntropyRefiner:
    def __init__(self, constraints: WaveformConstraints | None = None, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.rng = np.random.default_rng(seed)

    def refine(self, candidates: List[Waveform], env: Environment, objective: RobustObjective, budget: int = 80, elite_frac: float = 0.25) -> Tuple[Waveform, RobustResult, List[Dict[str, float]]]:
        history: List[Dict[str, float]] = []
        evaluated: List[Tuple[float, Waveform, RobustResult]] = []
        for w in candidates:
            wr = self.constraints.repair(w, env)
            r = objective.evaluate(wr, env)
            evaluated.append((r.robust_score, wr, r))
            history.append({"eval": float(objective.eval_count), "score": r.robust_score})
        bounds = self.constraints.bounds()
        if evaluated:
            X = np.vstack([w.to_vector() for _, w, _ in evaluated])
            scores = np.asarray([s for s, _, _ in evaluated])
            elite_idx = np.argsort(scores)[-max(1, min(len(scores), 4)):]
            mean = np.mean(X[elite_idx], axis=0)
            std = np.std(X[elite_idx], axis=0) + np.array([0.25, 40.0, 8.0, 3.0, 10.0, 0.8])
        else:
            mean = np.mean(bounds, axis=1)
            std = (bounds[:, 1] - bounds[:, 0]) / 3.0
        remaining = max(0, int(budget) - len(evaluated))
        while remaining > 0:
            m = min(10, remaining)
            samples = self.rng.normal(mean, std, size=(m, len(mean)))
            samples = np.clip(samples, bounds[:, 0], bounds[:, 1])
            for x in samples:
                w = self.constraints.repair(Waveform.from_vector(x), env)
                r = objective.evaluate(w, env)
                evaluated.append((r.robust_score, w, r))
                history.append({"eval": float(objective.eval_count), "score": r.robust_score})
            scores = np.asarray([s for s, _, _ in evaluated])
            X = np.vstack([w.to_vector() for _, w, _ in evaluated])
            k = max(2, int(np.ceil(elite_frac * len(evaluated))))
            elite_idx = np.argsort(scores)[-k:]
            mean = np.mean(X[elite_idx], axis=0)
            std = np.maximum(np.std(X[elite_idx], axis=0), np.array([0.05, 5.0, 1.0, 0.5, 2.0, 0.20])) * 0.88
            remaining -= m
        evaluated.sort(key=lambda t: t[0], reverse=True)
        return evaluated[0][1], evaluated[0][2], history


class RAGConformalRobustDesigner:
    def __init__(self, library: WaveformLibrary, constraints: WaveformConstraints | None = None, llm_client: LLMWaveformClient | None = None, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.library = library
        self.rule = RuleBasedDesigner(self.constraints, seed=seed)
        self.llm = llm_client or LLMWaveformClient(model="", api_url="", api_key="", use_api=False, require_api=False, constraints=self.constraints, seed=seed)
        self.refiner = CrossEntropyRefiner(self.constraints, seed=seed)

    def _dedup(self, candidates: List[Waveform], env: Environment) -> List[Waveform]:
        out, keys = [], set()
        for w in candidates:
            wr = self.constraints.repair(w, env)
            key = tuple(round(float(x), 2) for x in wr.to_vector())
            if key not in keys:
                keys.add(key)
                out.append(wr)
        return out

    def design(self, env: Environment, objective: RobustObjective, method_name: str = "rag_cra", n_retrieved: int = 8, n_llm_candidates: int = 6, local_budget: int = 80, use_retrieval: bool = True, use_rule: bool = True, use_llm: bool = True, use_refine: bool = True, use_robust: bool = True) -> DesignResult:
        t0 = time.perf_counter()
        retrieved = self.library.retrieve(env, k=n_retrieved) if use_retrieval else []
        candidates: List[Waveform] = []
        candidates.extend([e.waveform for e in retrieved])
        if use_rule:
            candidates.extend(self.rule.propose(env, n=max(2, n_llm_candidates // 2)))
        if use_llm:
            candidates.extend(self.llm.propose(env, retrieved, n=n_llm_candidates))
        if not candidates:
            candidates.extend(self.rule.propose(env, n=n_llm_candidates))
        dedup = self._dedup(candidates, env)

        selection_objective = objective
        if not use_robust:
            selection_objective = objective.__class__(simulator=objective.simulator, n_samples=1, cvar_alpha=objective.cvar_alpha, risk_pd_threshold=objective.risk_pd_threshold, seed=0)

        if use_refine:
            best_w, best_r, history = self.refiner.refine(dedup, env, selection_objective, budget=local_budget)
        else:
            best = None
            history = []
            for w in dedup:
                r = selection_objective.evaluate(w, env)
                if best is None or r.robust_score > best[0]:
                    best = (r.robust_score, w, r)
                history.append({"eval": float(selection_objective.eval_count), "score": best[0]})
            best_w, best_r = best[1], best[2]
        runtime = time.perf_counter() - t0
        extra = {
            "retrieved_count": len(retrieved),
            "initial_candidate_count": len(dedup),
            "selection_robust_score": best_r.robust_score,
            "selection_eval_count": selection_objective.eval_count,
            "llm_enabled": int(self.llm.use_api and use_llm),
            **self.llm.usage.to_dict(),
        }
        return DesignResult(method_name, best_w, best_r, runtime, selection_objective.eval_count, history, extra)


class RandomSearchOptimizer:
    def __init__(self, constraints: WaveformConstraints | None = None, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.rng = np.random.default_rng(seed)

    def design(self, env: Environment, objective: RobustObjective, budget: int = 100, method_name: str = "random") -> DesignResult:
        t0 = time.perf_counter()
        bounds = self.constraints.bounds()
        best = None
        history = []
        for _ in range(budget):
            x = self.rng.uniform(bounds[:, 0], bounds[:, 1])
            w = self.constraints.repair(Waveform.from_vector(x), env)
            r = objective.evaluate(w, env)
            if best is None or r.robust_score > best[0]:
                best = (r.robust_score, w, r)
            history.append({"eval": float(objective.eval_count), "score": best[0]})
        return DesignResult(method_name, best[1], best[2], time.perf_counter() - t0, objective.eval_count, history, {})


class ParticleSwarmOptimizer:
    def __init__(self, constraints: WaveformConstraints | None = None, n_particles: int = 24, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.n_particles = n_particles
        self.rng = np.random.default_rng(seed)

    def design(self, env: Environment, objective: RobustObjective, budget: int = 160, method_name: str = "pso") -> DesignResult:
        t0 = time.perf_counter()
        bounds = self.constraints.bounds()
        dim = bounds.shape[0]
        X = self.rng.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_particles, dim))
        V = np.zeros_like(X)
        pbest = X.copy(); pscore = np.full(self.n_particles, -np.inf)
        gbest = X[0].copy(); gscore = -np.inf; gbest_r = None
        history = []
        for _ in range(max(1, budget // self.n_particles)):
            for i in range(self.n_particles):
                w = self.constraints.repair(Waveform.from_vector(X[i]), env)
                r = objective.evaluate(w, env); s = r.robust_score
                if s > pscore[i]: pscore[i] = s; pbest[i] = X[i].copy()
                if s > gscore: gscore = s; gbest = X[i].copy(); gbest_r = r
                history.append({"eval": float(objective.eval_count), "score": gscore})
            V = 0.65 * V + 1.45 * self.rng.random(X.shape) * (pbest - X) + 1.45 * self.rng.random(X.shape) * (gbest - X)
            X = np.clip(X + V, bounds[:, 0], bounds[:, 1])
        best_w = self.constraints.repair(Waveform.from_vector(gbest), env)
        if gbest_r is None: gbest_r = objective.evaluate(best_w, env)
        return DesignResult(method_name, best_w, gbest_r, time.perf_counter() - t0, objective.eval_count, history, {})


class GeneticOptimizer:
    def __init__(self, constraints: WaveformConstraints | None = None, population: int = 32, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.population = population
        self.rng = np.random.default_rng(seed)

    def design(self, env: Environment, objective: RobustObjective, budget: int = 160, method_name: str = "ga") -> DesignResult:
        t0 = time.perf_counter()
        bounds = self.constraints.bounds()
        pop = self.rng.uniform(bounds[:, 0], bounds[:, 1], size=(self.population, bounds.shape[0]))
        best = None; history = []
        for _ in range(max(1, budget // self.population)):
            scores = []
            for x in pop:
                w = self.constraints.repair(Waveform.from_vector(x), env)
                r = objective.evaluate(w, env); scores.append(r.robust_score)
                if best is None or r.robust_score > best[0]: best = (r.robust_score, w, r)
                history.append({"eval": float(objective.eval_count), "score": best[0]})
            elites = pop[np.argsort(np.asarray(scores))[-max(2, self.population // 4):]]
            children = []
            while len(children) < self.population:
                p1, p2 = elites[self.rng.integers(len(elites))], elites[self.rng.integers(len(elites))]
                child = self.rng.random(pop.shape[1]) * p1 + (1 - self.rng.random(pop.shape[1])) * p2
                child += self.rng.normal(0.0, [0.12, 16, 3.5, 1.2, 8.0, 0.6], size=pop.shape[1])
                children.append(np.clip(child, bounds[:, 0], bounds[:, 1]))
            pop = np.asarray(children[:self.population])
        return DesignResult(method_name, best[1], best[2], time.perf_counter() - t0, objective.eval_count, history, {})


class DifferentialEvolutionOptimizer:
    def __init__(self, constraints: WaveformConstraints | None = None, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.seed = seed

    def design(self, env: Environment, objective: RobustObjective, budget: int = 160, method_name: str = "de") -> DesignResult:
        t0 = time.perf_counter(); bounds = self.constraints.bounds(); history = []
        if differential_evolution is None:
            return RandomSearchOptimizer(self.constraints, self.seed).design(env, objective, budget, method_name)
        best = {"score": -np.inf, "x": None, "r": None}
        def fn(x):
            w = self.constraints.repair(Waveform.from_vector(np.asarray(x)), env)
            r = objective.evaluate(w, env); s = r.robust_score
            if s > best["score"]: best.update(score=s, x=np.asarray(x).copy(), r=r)
            history.append({"eval": float(objective.eval_count), "score": best["score"]})
            return -s
        differential_evolution(fn, bounds=bounds.tolist(), seed=self.seed, polish=False, maxiter=max(1, budget // 18), popsize=3)
        best_w = self.constraints.repair(Waveform.from_vector(best["x"]), env)
        return DesignResult(method_name, best_w, best["r"], time.perf_counter() - t0, objective.eval_count, history, {})


class MLPolicyBaseline:
    def __init__(self, library: WaveformLibrary, constraints: WaveformConstraints | None = None, seed: int = 0):
        self.constraints = constraints or WaveformConstraints()
        self.library = library
        self.seed = seed
        self.rule = RuleBasedDesigner(self.constraints, seed=seed)
        self.model = None
        self._fit()

    def _fit(self):
        if RandomForestRegressor is None or MultiOutputRegressor is None or len(self.library.entries) < 8:
            return
        X = np.vstack([e.env.to_feature_vector() for e in self.library.entries])
        Y = np.vstack([e.waveform.to_vector() for e in self.library.entries])
        self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=120, random_state=self.seed, min_samples_leaf=2))
        self.model.fit(X, Y)

    def design(self, env: Environment, objective: RobustObjective, budget: int = 80, method_name: str = "ml_policy") -> DesignResult:
        t0 = time.perf_counter(); history = []
        if self.model is None:
            candidates = self.rule.propose(env, n=6)
        else:
            center = self.model.predict(env.to_feature_vector().reshape(1, -1))[0]
            bounds = self.constraints.bounds()
            rng = np.random.default_rng(self.seed + int(abs(env.snr_db) * 10))
            candidates = []
            for _ in range(max(6, budget // 10)):
                vec = center + rng.normal(0.0, [0.2, 18.0, 5.0, 1.5, 8.0, 0.5])
                candidates.append(self.constraints.repair(Waveform.from_vector(np.clip(vec, bounds[:, 0], bounds[:, 1])), env))
        best = None
        for w in candidates:
            r = objective.evaluate(w, env)
            if best is None or r.robust_score > best[0]: best = (r.robust_score, w, r)
            history.append({"eval": float(objective.eval_count), "score": best[0]})
        return DesignResult(method_name, best[1], best[2], time.perf_counter() - t0, objective.eval_count, history, {"model_type": "rf_multioutput" if self.model is not None else "rule_fallback"})


def make_designer(method: str, library: WaveformLibrary, constraints: WaveformConstraints, llm_client: LLMWaveformClient, seed: int):
    if method in {"rag_cra", "direct_llm", "rag_only", "rag_cra_no_api", "rag_cra_no_refine", "rag_cra_no_robust"}:
        return RAGConformalRobustDesigner(library, constraints, llm_client=llm_client, seed=seed)
    if method == "ml_policy": return MLPolicyBaseline(library, constraints, seed)
    if method == "random": return RandomSearchOptimizer(constraints, seed)
    if method == "pso": return ParticleSwarmOptimizer(constraints, 24, seed)
    if method == "ga": return GeneticOptimizer(constraints, 32, seed)
    if method == "de": return DifferentialEvolutionOptimizer(constraints, seed)
    raise ValueError(f"Unknown method: {method}")


def run_method(method: str, designer, env: Environment, objective: RobustObjective, budget: int, local_budget: int, llm_candidates: int) -> DesignResult:
    if method == "rag_cra":
        return designer.design(env, objective, method_name=method, use_retrieval=True, use_rule=True, use_llm=True, use_refine=True, use_robust=True, local_budget=local_budget, n_llm_candidates=llm_candidates)
    if method == "direct_llm":
        return designer.design(env, objective, method_name=method, use_retrieval=False, use_rule=False, use_llm=True, use_refine=False, use_robust=False, local_budget=local_budget, n_llm_candidates=llm_candidates)
    if method == "rag_only":
        return designer.design(env, objective, method_name=method, use_retrieval=True, use_rule=False, use_llm=False, use_refine=False, use_robust=False, local_budget=local_budget, n_llm_candidates=llm_candidates)
    if method == "rag_cra_no_api":
        # v4 clean ablation: no API means no LLM proposal call and no rule-fallback
        # candidates injected through the LLM client. The method is retrieval + rule +
        # the same robust refiner, so any gap to rag_cra is attributable to parsed
        # LLM candidates rather than a fallback side effect.
        return designer.design(env, objective, method_name=method, use_retrieval=True, use_rule=True, use_llm=False, use_refine=True, use_robust=True, local_budget=local_budget, n_llm_candidates=llm_candidates)
    if method == "rag_cra_no_refine":
        return designer.design(env, objective, method_name=method, use_retrieval=True, use_rule=True, use_llm=True, use_refine=False, use_robust=True, local_budget=local_budget, n_llm_candidates=llm_candidates)
    if method == "rag_cra_no_robust":
        return designer.design(env, objective, method_name=method, use_retrieval=True, use_rule=True, use_llm=True, use_refine=True, use_robust=False, local_budget=local_budget, n_llm_candidates=llm_candidates)
    return designer.design(env, objective, budget=budget, method_name=method)
