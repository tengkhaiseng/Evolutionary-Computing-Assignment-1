"""
JIE42903 Evolutionary Computing - Assignment
TV program scheduling with a genetic algorithm, with a Streamlit interface.

Part 1 is the Week 4 tutorial code ("TV Scheduling - Genetic Algorithm").
The genetic-algorithm logic is unchanged. Only these things differ:

  * file_path points to the modified CSV next to this file instead of the
    Google Colab path '/content/program_ratings.csv'.
  * MUT_R defaults to 0.02 because the assignment limits the mutation rate
    to 0.01 - 0.05 (the tutorial used 0.2).
  * The brute-force search and the final GA run are wrapped in two functions
    (brute_force_best_schedule, run_schedule) so the interface can call them
    with the user's CO_R and MUT_R. The brute-force result is cached.
  * The tutorial's print() output is replaced by Streamlit tables.

Part 2 is the Streamlit interface.

Run locally:   streamlit run app.py
"""
import csv
import os
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TV Scheduling GA", page_icon=":tv:", layout="wide")


# ==============================================================================
# PART 1 - GENETIC ALGORITHM (WEEK 4 TUTORIAL CODE)
# ==============================================================================

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings


# Path to the CSV file (the modified ratings, stored next to this script)
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'program_ratings.csv')

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)


##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

GEN = 100
POP = 50
CO_R = 0.8
MUT_R = 0.02  # tutorial: 0.2; assignment range for the mutation rate is 0.01 - 0.05
EL_S = 2

all_programs = list(ratings.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots


######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating


# initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules


# selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule


# Does not depend on CO_R / MUT_R, so it is computed once and reused (cached).
@st.cache_data(show_spinner="Brute-force search over every order of the 10 programs (first run only)...")
def brute_force_best_schedule():
    # calling the pop func.
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)

    # callin the schedule func.
    best_schedule = finding_best_schedule(all_possible_schedules)
    return best_schedule


############################################# GENETIC ALGORITHM #############################################################################

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2


# mutating
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule


# calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)


# genetic algorithms with parameters
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitsm
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]


##################################################### RESULTS ###################################################################################

def run_schedule(initial_best_schedule, crossover_rate=CO_R, mutation_rate=MUT_R):
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    genetic_schedule = genetic_algorithm(list(initial_best_schedule), generations=GEN, population_size=POP,
                                         crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                                         elitism_size=EL_S)

    final_schedule = list(initial_best_schedule) + genetic_schedule[:rem_t_slots]
    return final_schedule


# ==============================================================================
# PART 2 - STREAMLIT INTERFACE
# ==============================================================================

NUM_TRIALS = 3

# Parameter limits from the assignment brief
CO_R_MIN, CO_R_MAX, CO_R_DEFAULT = 0.0, 0.95, 0.8
MUT_R_MIN, MUT_R_MAX, MUT_R_DEFAULT = 0.01, 0.05, 0.02


def schedule_to_table(schedule, brute_force_len):
    rows = []
    for slot, program in enumerate(schedule):
        rows.append({
            "Time Slot": f"{all_time_slots[slot]:02d}:00",
            "Program": program,
            "Rating": ratings[program][slot],
            "Assigned by": "Brute force" if slot < brute_force_len else "Genetic algorithm",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- header
st.title(":tv: TV Program Scheduling with a Genetic Algorithm")
st.write(
    "Choose a crossover rate (CO_R) and mutation rate (MUT_R) for each of the three trials, "
    "then run the genetic algorithm. Each trial produces an 18-hour schedule (06:00 – 23:00) "
    "that tries to maximise the total audience rating."
)

with st.expander("Program ratings dataset (modified CSV)"):
    ratings_df = pd.DataFrame(ratings, index=[f"{h:02d}:00" for h in all_time_slots]).T
    ratings_df.index.name = "Program"
    st.dataframe(ratings_df.style.format("{:.1f}"), width="stretch")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Fixed GA settings")
    st.write(f"Generations (GEN): **{GEN}**")
    st.write(f"Population size (POP): **{POP}**")
    st.write(f"Elitism size (EL_S): **{EL_S}**")
    st.divider()
    st.header("Reproducibility")
    use_seed = st.checkbox("Use a fixed random seed", value=True,
                           help="With a fixed seed every trial starts from the same random numbers, "
                                "so differences between trials come from CO_R and MUT_R only.")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1, disabled=not use_seed)

# ----------------------------------------------------------------------------- parameter input
st.subheader("1. Set the parameters for each trial")
param_cols = st.columns(NUM_TRIALS)
params = []
for i, col in enumerate(param_cols, start=1):
    with col:
        with st.container(border=True):
            st.markdown(f"**Trial {i}**")
            co_r = st.slider("Crossover rate (CO_R)", CO_R_MIN, CO_R_MAX, CO_R_DEFAULT, 0.01,
                             key=f"co_r_{i}", help="Range 0.00 – 0.95, default 0.80")
            mut_r = st.slider("Mutation rate (MUT_R)", MUT_R_MIN, MUT_R_MAX, MUT_R_DEFAULT, 0.01,
                              key=f"mut_r_{i}", help="Range 0.01 – 0.05, default 0.02")
            params.append({"Trial": i, "CO_R": round(co_r, 2), "MUT_R": round(mut_r, 2)})

run = st.button("Run genetic algorithm for all 3 trials", type="primary", width="stretch")

# ----------------------------------------------------------------------------- run
if run:
    initial_best_schedule = brute_force_best_schedule()
    results = []
    with st.spinner("Running the genetic algorithm..."):
        for p in params:
            if use_seed:
                random.seed(int(seed))
            final_schedule = run_schedule(initial_best_schedule, crossover_rate=p["CO_R"], mutation_rate=p["MUT_R"])
            results.append({**p, "schedule": final_schedule, "total": fitness_function(final_schedule)})
    st.session_state["results"] = results
    st.session_state["brute_force"] = initial_best_schedule
    st.session_state["seed_used"] = int(seed) if use_seed else None

# ----------------------------------------------------------------------------- results
if "results" in st.session_state:
    results = st.session_state["results"]
    brute_force = st.session_state["brute_force"]
    bf_len = len(brute_force)

    if [{k: r[k] for k in ("Trial", "CO_R", "MUT_R")} for r in results] != params:
        st.warning("The parameters have changed since the last run. Click the button again to update the results.")

    st.subheader("2. Results")
    seed_used = st.session_state["seed_used"]
    st.caption(f"Random seed: {seed_used if seed_used is not None else 'not fixed'} · "
               f"06:00–{all_time_slots[bf_len - 1]:02d}:00 comes from the brute-force search, "
               f"{all_time_slots[bf_len]:02d}:00–23:00 from the genetic algorithm.")

    summary = pd.DataFrame([{"Trial": f"Trial {r['Trial']}", "CO_R": r["CO_R"], "MUT_R": r["MUT_R"],
                             "Total Rating": round(r["total"], 2)} for r in results])
    best = summary["Total Rating"].max()

    metric_cols = st.columns(NUM_TRIALS)
    for col, r in zip(metric_cols, results):
        col.markdown(f"**Trial {r['Trial']}** · CO_R = {r['CO_R']:.2f} · MUT_R = {r['MUT_R']:.2f}")
        col.metric("Total rating", f"{r['total']:.2f}",
                   "highest" if round(r["total"], 2) == best else None, delta_color="off")

    tabs = st.tabs([f"Trial {r['Trial']}" for r in results] + ["Side-by-side comparison"])
    for tab, r in zip(tabs, results):
        with tab:
            st.markdown(f"**Parameters:** CO_R = {r['CO_R']:.2f}, MUT_R = {r['MUT_R']:.2f} · "
                        f"**Total rating:** {r['total']:.2f}")
            table = schedule_to_table(r["schedule"], bf_len)
            st.dataframe(table, hide_index=True, width="stretch", height=(len(table) + 1) * 35 + 3)

    with tabs[-1]:
        compare = pd.DataFrame({"Time Slot": [f"{h:02d}:00" for h in all_time_slots]})
        for r in results:
            compare[f"Trial {r['Trial']} (CO_R={r['CO_R']:.2f}, MUT_R={r['MUT_R']:.2f})"] = r["schedule"]
        st.dataframe(compare, hide_index=True, width="stretch", height=(len(compare) + 1) * 35 + 3)
        st.dataframe(summary, hide_index=True, width="stretch")
        st.bar_chart(summary, x="Trial", y="Total Rating")
else:
    st.info("Set the parameters above and click **Run genetic algorithm for all 3 trials**.")
