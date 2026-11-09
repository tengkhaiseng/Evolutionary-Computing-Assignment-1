"""
Reproduce the experiments in the report.

  python experiments.py

1. The three trials, each run once with seed 42 (the app's default settings).
2. A robustness check: each trial setting repeated with seeds 0-29.
3. A sweep over CO_R x MUT_R (30 seeds per cell) to show the overall trend.

Results are written to the results/ folder as CSV files.
"""

import csv
import statistics
from pathlib import Path

import ga_scheduler as ga

TRIALS = [
    ("Trial 1", 0.80, 0.20),  # default values
    ("Trial 2", 0.95, 0.05),
    ("Trial 3", 0.30, 0.01),
]
SEED = 42
N_SEEDS = 30
SWEEP_CO = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
SWEEP_MUT = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20]

OUT = Path(__file__).parent / "results"


def run(ratings, co_r, mut_r, seed):
    return ga.genetic_algorithm(ratings, crossover_rate=co_r, mutation_rate=mut_r, seed=seed)


def main():
    OUT.mkdir(exist_ok=True)
    ratings, hours = ga.read_csv_to_dict(Path(__file__).parent / "program_ratings.csv")
    opt_schedule, opt_value = ga.optimal_schedule(ratings)
    print(f"Exact optimum (benchmark): {opt_value:.1f}\n")

    # 1. single runs
    schedules, convergence, summary = {}, {}, []
    for name, co, mu in TRIALS:
        res = run(ratings, co, mu, SEED)
        schedules[name] = res.best_schedule
        convergence[name] = res.best_history
        convergence[name + " (avg)"] = res.avg_history
        fits = [run(ratings, co, mu, s) for s in range(N_SEEDS)]
        values = [r.best_fitness for r in fits]
        summary.append({
            "trial": name, "co_r": co, "mut_r": mu,
            "total_rating": round(res.best_fitness, 2),
            "found_at_generation": res.found_at_generation,
            "gap_to_optimum": round(opt_value - res.best_fitness, 2),
            "mean_30": round(statistics.mean(values), 3),
            "sd_30": round(statistics.pstdev(values), 3),
            "best_30": max(values), "worst_30": min(values),
            "hit_optimum_30": sum(abs(v - opt_value) < 1e-9 for v in values),
            "mean_found_gen_30": round(statistics.mean(r.found_at_generation for r in fits), 1),
        })
        print(f"{name}: CO_R={co:.2f} MUT_R={mu:.2f} -> total {res.best_fitness:.1f} "
              f"(found at gen {res.found_at_generation}); mean over {N_SEEDS} seeds "
              f"{statistics.mean(values):.3f} +/- {statistics.pstdev(values):.3f}")

    with open(OUT / "trial_schedules.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_slot"] + [n for n, _, _ in TRIALS] + ["exact_optimum"])
        for i, h in enumerate(hours):
            w.writerow([f"{h:02d}:00"] + [schedules[n][i] for n, _, _ in TRIALS] + [opt_schedule[i]])

    with open(OUT / "trial_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    with open(OUT / "trial_convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation"] + list(convergence))
        for g in range(len(next(iter(convergence.values())))):
            w.writerow([g] + [round(v[g], 4) for v in convergence.values()])

    # 3. sweep
    print("\nSweep: mean best total rating over", N_SEEDS, "seeds")
    print("CO_R \\ MUT_R " + " ".join(f"{m:>6.2f}" for m in SWEEP_MUT))
    with open(OUT / "parameter_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["co_r", "mut_r", "mean", "sd", "hit_optimum", "mean_found_gen"])
        for co in SWEEP_CO:
            line = []
            for mu in SWEEP_MUT:
                rs = [run(ratings, co, mu, s) for s in range(N_SEEDS)]
                vals = [r.best_fitness for r in rs]
                m = statistics.mean(vals)
                line.append(m)
                w.writerow([co, mu, round(m, 3), round(statistics.pstdev(vals), 3),
                            sum(abs(v - opt_value) < 1e-9 for v in vals),
                            round(statistics.mean(r.found_at_generation for r in rs), 1)])
            print(f"{co:>12.2f} " + " ".join(f"{v:>6.2f}" for v in line))
    print(f"\nCSV files written to {OUT}")


if __name__ == "__main__":
    main()
