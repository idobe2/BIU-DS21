import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Step 1: Define the states ---
STATES = [
    "Start",
    "Electricity Issue",
    "Router Issue",
    "Connection Issue",
    "General Issue",
    "Issue Resolved"
]

# --- Step 2: Define the Markov Transition Matrix ---
TRANSITION_MATRIX = {
    "Start": {
      "Electricity Issue": 1.0
    },
    "Electricity Issue": {
        "Router Issue": 0.7,
        "Issue Resolved": 0.3
    },
    "Router Issue": {
        "Connection Issue": 0.6,
        "Issue Resolved": 0.4
    },
    "Connection Issue": {
        "General Issue": 0.2, 
        "Issue Resolved": 0.8
    },
    "General Issue": {},
    "Issue Resolved": {}
}

# --- Step 3: Simulate the Diagnostic Process ---
def run_diagnostic(transition_matrix, start_state="Start"):
    current_state = start_state
    path = [current_state]

    while current_state in transition_matrix and transition_matrix[current_state]:
        next_states = transition_matrix[current_state]
        states = list(next_states.keys())
        probabilities = list(next_states.values())

        current_state = random.choices(states, probabilities)[0]
        path.append(current_state)

    return (
        current_state,
        path
    )

# print("--- Running Single Simulation ---")
# final_state, diagnostic_path = run_diagnostic(TRANSITION_MATRIX)
# print(f"Simulation ended in: {final_state}")
# print(f"Full path was: {' -> '.join(diagnostic_path)}")

# --- Step 4: Run Simulations and Analyze ---
N = 1000
total_path_length = 0
end_state_counts = {
    "Issue Resolved": 0,
    "General Issue": 0
}
total_state_visits = {}
for state in TRANSITION_MATRIX.keys():
    total_state_visits[state] = 0

for i in range(N):
    final_state, path = run_diagnostic(TRANSITION_MATRIX)
    total_path_length += len(path)
    
    if final_state in end_state_counts:
        end_state_counts[final_state] += 1
    
    for state in path:
        if state in total_state_visits:
            total_state_visits[state] += 1

print(f"\n--- Analysis of {N} Simulations ---")

avg_length = total_path_length / N
print(f"Average diagnostic path length: {round(avg_length, 2)} steps")

success_rate = (end_state_counts["Issue Resolved"] / N) * 100
failure_rate = (end_state_counts["General Issue"] / N) * 100
print(f"Success rate ('Issue Resolved'): {round(success_rate, 2)}%")
print(f"Failure rate ('General Issue'): {round(failure_rate, 2)}%")

total_visits = sum(total_state_visits.values())
print("\nPercentage of time spent in each state:")
most_stuck_state = ""
max_visits_pct = 0

for state, count in total_state_visits.items():
    pct = (count / total_visits) * 100
    print(f"- {state}: {round(pct, 2)}%")
    if state not in ["Start", "Issue Resolved", "General Issue"]:
        if pct > max_visits_pct:
            max_visits_pct = pct
            most_stuck_state = state

print(f"\nMost common diagnostic step (bottleneck): {most_stuck_state}")


# --- Step 5: Visual Analysis ---
print("\n--- Generating Markov Chain Graph ---")

G = nx.DiGraph()

for source, targets in TRANSITION_MATRIX.items():
    G.add_node(source) 
    for target, prob in targets.items():
        G.add_edge(source, target, label=f"{prob:.2f}")

pos = nx.spring_layout(G, seed=42, k=0.9) 

plt.figure(figsize=(14, 10))

nx.draw(G, pos, with_labels=True, node_color="lightblue", 
        node_size=3000, font_size=9, font_weight="bold", arrows=True,
        arrowstyle='->', arrowsize=20)

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                            font_color='red', font_size=9)

plt.title("Markov Chain: Diagnostic Process", size=16)
plt.axis('off')
plt.tight_layout()

output_filename = 'diagnostic_chain.png'
plt.savefig(output_filename)

print(f"Graph saved to {output_filename}")