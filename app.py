"""
Streamlit interface for the TV program scheduling Genetic Algorithm.
JIE42903 Evolutionary Computing - Assignment 1

Run locally:  streamlit run app.py
"""

from pathlib import Path
import statistics

import altair as alt
import pandas as pd
import streamlit as st

import ga_scheduler as ga

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "program_ratings.csv"
ORIGINAL_FILE = BASE_DIR / "program_ratings_original.csv"

# co_step 0.05 suits Trial 1's 0-0.95 range; Trials 2 and 3 start at 0.01, so they need a 0.01 step to reach 0.95.
TRIAL_DEFAULTS = [
    {"co_r": ga.CO_R, "mut_r": ga.MUT_R, "co_range": ga.DEFAULT_TRIAL_CO_R_RANGE, "co_step": 0.05,
     "mut_range": ga.DEFAULT_TRIAL_MUT_R_RANGE, "label": "Trial 1 (default values)"},  # CO_R 0.8, MUT_R 0.2
    {"co_r": 0.95, "mut_r": 0.05, "co_range": ga.CO_R_RANGE, "co_step": 0.01,
     "mut_range": ga.MUT_R_RANGE, "label": "Trial 2"},                                  # upper limits
    {"co_r": 0.30, "mut_r": 0.01, "co_range": ga.CO_R_RANGE, "co_step": 0.01,
     "mut_range": ga.MUT_R_RANGE, "label": "Trial 3"},                                  # low rates
]

st.set_page_config(page_title="TV Scheduling GA", page_icon="📺", layout="wide")


# ----------------------------------------------------------------- helpers
@st.cache_data
def load_ratings_from_path(path: str):
    return ga.read_csv_to_dict(path)


def ratings_dataframe(ratings, hours):
    df = pd.DataFrame(ratings, index=[f"{h:02d}:00" for h in hours]).T
    df.index.name = "Program"
    return df


def run_trial(ratings, co_r, mut_r, gen, pop, elit, seed):
    return ga.genetic_algorithm(
        ratings,
        generations=gen,
        population_size=pop,
        crossover_rate=co_r,
        mutation_rate=mut_r,
        elitism_size=elit,
        seed=seed,
    )


def schedule_table(schedule, hours, ratings):
    return pd.DataFrame(ga.schedule_rows(schedule, hours, ratings))


def convergence_chart(series: dict[str, list[float]], y_title: str, height: int = 320):
    """Line chart of fitness per generation; the y-axis is not forced to start at zero."""
    long = pd.DataFrame(
        [{"Generation": g, "Series": name, y_title: v}
         for name, values in series.items() for g, v in enumerate(values)]
    )
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Generation:Q"),
            y=alt.Y(f"{y_title}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", sort=list(series), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Series", "Generation", alt.Tooltip(f"{y_title}:Q", format=".2f")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


RATING_COL = st.column_config.NumberColumn("Rating", format="%.1f")


# ----------------------------------------------------------------- header
st.title("📺 TV Program Scheduling with a Genetic Algorithm")
st.write(
    "Find a daily TV schedule from **06:00 to 23:00** that maximises total audience rating. "
    "Each hour holds one program and every program must air at least once. "
    "Set the **crossover rate (CO_R)** and **mutation rate (MUT_R)** for three trials, run them, "
    "then compare the schedules."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Ratings CSV (optional)", type="csv",
                                help="Leave empty to use the modified program_ratings.csv from the repository.")
    st.header("GA settings")
    st.caption("Tutorial defaults: 100 generations, population 50, elitism 2.")
    generations = st.number_input("Generations (GEN)", 10, 1000, ga.GEN, step=10)
    population = st.number_input("Population size (POP)", 10, 500, ga.POP, step=10)
    elitism = st.number_input("Elitism size (EL_S)", 0, 10, ga.EL_S)
    fixed_seed = st.checkbox("Use a fixed random seed", value=True,
                             help="With a fixed seed every trial uses the same random numbers, so differences "
                                  "come only from CO_R and MUT_R, and results can be reproduced.")
    seed = st.number_input("Random seed", 0, 10_000, 42, disabled=not fixed_seed)
    st.header("Robustness check")
    repeat_runs = st.checkbox("Also repeat each trial over many seeds", value=False,
                              help="A single GA run depends on luck. This reruns each trial with different "
                                   "seeds and reports the average.")
    n_repeats = st.slider("Number of runs per trial", 5, 50, 30, disabled=not repeat_runs)

# ----------------------------------------------------------------- data
try:
    if uploaded is not None:
        ratings, hours = ga.read_csv_to_dict(uploaded)
        data_label = uploaded.name
    else:
        ratings, hours = load_ratings_from_path(str(DATA_FILE))
        data_label = DATA_FILE.name
except Exception as exc:  # bad upload
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

programs = list(ratings.keys())
optimum = ga.optimal_schedule(ratings)

with st.expander(f"📊 Ratings data ({data_label}: {len(programs)} programs × {len(hours)} hours)"):
    df_ratings = ratings_dataframe(ratings, hours)
    changed = set()
    if uploaded is None and ORIGINAL_FILE.exists():
        original, _ = load_ratings_from_path(str(ORIGINAL_FILE))
        changed = {(p, i) for p in ratings for i in range(len(hours))
                   if p in original and abs(ratings[p][i] - original[p][i]) > 1e-9}

    def highlight(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for p, i in changed:
            styles.iloc[data.index.get_loc(p), i] = "background-color: #ffe08a; color: #000; font-weight: 600"
        return styles

    hour_cols = {c: st.column_config.NumberColumn(c, width=42) for c in df_ratings.columns}
    st.dataframe(df_ratings.style.format("{:.1f}").apply(highlight, axis=None), width="stretch",
                 column_config={"Program": st.column_config.TextColumn("Program", width=105), **hour_cols})
    if changed:
        st.caption(f"Highlighted cells: the {len(changed)} ratings changed from the original tutorial CSV.")
        rows = [{"Program": p, "Hour": f"{hours[i]:02d}:00", "Original": original[p][i], "Modified": ratings[p][i]}
                for p, i in sorted(changed, key=lambda x: (programs.index(x[0]), x[1]))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="content", height=35 * (len(rows) + 1) + 3,
                     column_config={"Original": st.column_config.NumberColumn(format="%.1f"),
                                    "Modified": st.column_config.NumberColumn(format="%.1f")})

# ----------------------------------------------------------------- trial inputs
st.subheader("1. Set the parameters for each trial")
st.caption(f"Trial 1 starts at the default values (CO_R = {ga.CO_R:.1f}, MUT_R = {ga.MUT_R:.1f}). "
           f"Trials 2 and 3 use the required ranges: CO_R {ga.CO_R_RANGE[0]}–{ga.CO_R_RANGE[1]}, "
           f"MUT_R {ga.MUT_R_RANGE[0]}–{ga.MUT_R_RANGE[1]}.")

with st.form("trial_form"):
    cols = st.columns(3)
    trial_params = []
    for i, col in enumerate(cols):
        with col:
            d = TRIAL_DEFAULTS[i]
            st.markdown(f"**{d['label']}**")
            co_r = st.slider("Crossover rate (CO_R)", d["co_range"][0], d["co_range"][1],
                             d["co_r"], d["co_step"], key=f"co_{i}")
            mut_r = st.slider("Mutation rate (MUT_R)", d["mut_range"][0], d["mut_range"][1],
                              d["mut_r"], 0.01, key=f"mut_{i}")
            trial_params.append({"co_r": co_r, "mut_r": mut_r})
    submitted = st.form_submit_button("▶ Run genetic algorithm", type="primary", width="stretch")

if submitted:
    run_seed = int(seed) if fixed_seed else None
    with st.spinner("Evolving schedules..."):
        results = []
        for p in trial_params:
            res = run_trial(ratings, p["co_r"], p["mut_r"], int(generations), int(population), int(elitism), run_seed)
            entry = {"params": p, "result": res}
            if repeat_runs:
                fits = [run_trial(ratings, p["co_r"], p["mut_r"], int(generations), int(population),
                                  int(elitism), s).best_fitness for s in range(int(n_repeats))]
                entry["repeats"] = fits
            results.append(entry)
    st.session_state["results"] = results
    st.session_state["settings"] = {"generations": int(generations), "population": int(population),
                                    "elitism": int(elitism), "seed": run_seed, "data": data_label}

# ----------------------------------------------------------------- results
results = st.session_state.get("results")
if not results:
    st.info("Set the parameters above and press **Run genetic algorithm**.")
    st.stop()

settings = st.session_state["settings"]
st.subheader("2. Results")
st.caption(f"Data: {settings['data']} · GEN = {settings['generations']} · POP = {settings['population']} · "
           f"EL_S = {settings['elitism']} · seed = {settings['seed'] if settings['seed'] is not None else 'random'}")

tab_names = [f"Trial {i + 1}" for i in range(len(results))] + ["Comparison"]
tabs = st.tabs(tab_names)

for i, (tab, entry) in enumerate(zip(tabs, results)):
    res, p = entry["result"], entry["params"]
    with tab:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CO_R", f"{p['co_r']:.2f}")
        m2.metric("MUT_R", f"{p['mut_r']:.2f}")
        m3.metric("Total rating", f"{res.best_fitness:.1f}")
        if optimum:
            m4.metric("Share of optimum", f"{100 * res.best_fitness / optimum[1]:.1f}%",
                      delta=f"{res.best_fitness - optimum[1]:+.1f} vs optimum {optimum[1]:.1f}",
                      delta_color="off" if abs(res.best_fitness - optimum[1]) < 1e-9 else "normal")
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown("**Schedule**")
            st.dataframe(schedule_table(res.best_schedule, hours, ratings), hide_index=True,
                         width="stretch", height=670, column_config={"Rating": RATING_COL})
        with right:
            st.markdown("**Convergence**")
            convergence_chart({"Best in population": res.best_history,
                               "Population average": res.avg_history}, "Total rating")
            st.caption(f"Best schedule first found in generation {res.found_at_generation}.")
            counts = pd.Series(res.best_schedule).value_counts().rename_axis("Program").reset_index(name="Hours aired")
            st.markdown("**Air time per program**")
            st.dataframe(counts, hide_index=True, width="stretch")
            if "repeats" in entry:
                fits = entry["repeats"]
                st.markdown(f"**Robustness over {len(fits)} seeds**")
                r1, r2, r3 = st.columns(3)
                r1.metric("Mean", f"{statistics.mean(fits):.2f}")
                r2.metric("Std dev", f"{statistics.pstdev(fits):.2f}")
                r3.metric("Best / worst", f"{max(fits):.1f} / {min(fits):.1f}")

with tabs[-1]:
    summary = []
    for i, entry in enumerate(results):
        res, p = entry["result"], entry["params"]
        row = {"Trial": f"Trial {i + 1}", "CO_R": p["co_r"], "MUT_R": p["mut_r"],
               "Total rating": res.best_fitness, "Found at generation": res.found_at_generation,
               "Distinct programs": len(set(res.best_schedule))}
        if optimum:
            row["Gap to optimum"] = round(optimum[1] - res.best_fitness, 2)
        if "repeats" in entry:
            row["Mean over seeds"] = round(statistics.mean(entry["repeats"]), 3)
            row["Std dev over seeds"] = round(statistics.pstdev(entry["repeats"]), 3)
        summary.append(row)
    st.markdown("**Summary**")
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch",
                 column_config={"CO_R": st.column_config.NumberColumn(format="%.2f"),
                                "MUT_R": st.column_config.NumberColumn(format="%.2f"),
                                "Total rating": st.column_config.NumberColumn(format="%.1f")})

    top = max(e["result"].best_fitness for e in results)
    winners = [i for i, e in enumerate(results) if abs(e["result"].best_fitness - top) < 1e-9]
    if len(winners) == 1:
        w = winners[0]
        st.success(f"Highest total rating: **Trial {w + 1}** (CO_R = {results[w]['params']['co_r']:.2f}, "
                   f"MUT_R = {results[w]['params']['mut_r']:.2f}) with {top:.1f}.")
    else:
        fastest = min(winners, key=lambda i: results[i]["result"].found_at_generation)
        names = " and ".join(f"Trial {i + 1}" for i in winners)
        st.success(f"{names} tie on the highest total rating ({top:.1f}). Trial {fastest + 1} reached it sooner "
                   f"(generation {results[fastest]['result'].found_at_generation}).")

    st.markdown("**Schedules side by side**")
    side = pd.DataFrame({"Time Slot": [f"{h:02d}:00" for h in hours]})
    for i, entry in enumerate(results):
        side[f"Trial {i + 1}"] = entry["result"].best_schedule
    if optimum:
        side["Exact optimum (benchmark)"] = optimum[0]
    st.dataframe(side, hide_index=True, width="stretch", height=670)

    st.markdown("**Convergence of the three trials**")
    convergence_chart({f"Trial {i + 1}": e["result"].best_history for i, e in enumerate(results)},
                      "Best total rating", height=340)

    if optimum:
        st.caption(f"The exact optimum ({optimum[1]:.1f}) is computed with a linear assignment solver and is "
                   "shown only as a benchmark. It is not used by the GA.")
