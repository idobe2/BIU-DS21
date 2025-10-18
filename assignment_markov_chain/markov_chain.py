import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Step 1: Define the states ---
# The possible states in the diagnostic process.
STATES = [
    "Electricity Issue",
    "Router Issue",
    "Physical Connection Issue",
    "General Outage",
    "Issue Resolved"
]

# The final state, which is an absorbing state.
ABSORBING_STATE = "Issue Resolved"

# --- Step 2: Define the Markov Transition Matrix ---
# A dictionary where each key is a state, and the value is another dictionary
# representing the probabilities of transitioning to other states.
TRANSITION_PROBS = {
    "Electricity Issue": {
        "Electricity Issue": 0.1, "Router Issue": 0.3, "Physical Connection Issue": 0.2, "General Outage": 0.3, "Issue Resolved": 0.1
    },
    "Router Issue": {
        "Electricity Issue": 0.2, "Router Issue": 0.2, "Physical Connection Issue": 0.3, "General Outage": 0.2, "Issue Resolved": 0.1
    },
    "Physical Connection Issue": {
        "Electricity Issue": 0.3, "Router Issue": 0.2, "Physical Connection Issue": 0.2, "General Outage": 0.2, "Issue Resolved": 0.1
    },
    "General Outage": {
        "Electricity Issue": 0.4, "Router Issue": 0.2, "Physical Connection Issue": 0.2, "General Outage": 0.1, "Issue Resolved": 0.1
    },
    "Issue Resolved": {
        "Electricity Issue": 0.0, "Router Issue": 0.0, "Physical Connection Issue": 0.0, "General Outage": 0.0, "Issue Resolved": 1.0
    }
}

# For easier calculations, we convert the dictionary to a NumPy matrix.
# The order is based on the STATES list.
TRANSITION_MATRIX = np.array([
    list(TRANSITION_PROBS["Electricity Issue"].values()),
    list(TRANSITION_PROBS["Router Issue"].values()),
    list(TRANSITION_PROBS["Physical Connection Issue"].values()),
    list(TRANSITION_PROBS["General Outage"].values()),
    list(TRANSITION_PROBS["Issue Resolved"].values()),
])


# --- Step 3: Write a simulation of the diagnostic process ---
def run_single_simulation(start_state, max_steps=50):
    """
    Runs a single simulation of the diagnostic process.

    Args:
        start_state (str): The initial state of the simulation.
        max_steps (int): The maximum number of steps before considering the process failed.

    Returns:
        tuple: A tuple containing:
            - list[str]: The history of states visited.
            - bool: True if the issue was resolved, False otherwise.
    """
    current_state = start_state
    history = [current_state]
    steps = 0

    while current_state != ABSORBING_STATE and steps < max_steps:
        current_state_index = STATES.index(current_state)
        probabilities = TRANSITION_MATRIX[current_state_index]

        # Choose the next state based on the transition probabilities
        next_state = np.random.choice(STATES, p=probabilities)
        history.append(next_state)
        current_state = next_state
        steps += 1

    # Check if the simulation ended by resolving the issue
    is_success = (current_state == ABSORBING_STATE)
    return history, is_success


# --- Step 4: Run the simulation and experiments ---
def run_experiments(num_simulations=1000):
    """
    Performs a given number of simulations and calculates key statistics.
    """
    print(f"--- Running {num_simulations} Simulations ---")

    total_steps_for_success = 0
    success_count = 0
    # Dictionary to count how many times each state is visited during the process
    state_visit_counts = {state: 0 for state in STATES if state != ABSORBING_STATE}

    for _ in range(num_simulations):
        # All simulations start from the first state
        start_state = STATES[0]
        history, was_successful = run_single_simulation(start_state)

        if was_successful:
            success_count += 1
            # Add the number of steps (path length - 1)
            total_steps_for_success += len(history) - 1
            # Count each intermediate state visited
            for state in history[:-1]:  # Exclude the final "Resolved" state
                if state in state_visit_counts:
                    state_visit_counts[state] += 1

    # Calculate statistics
    average_steps = total_steps_for_success / success_count if success_count > 0 else 0
    success_rate = (success_count / num_simulations) * 100
    # Find the state where customers spend the most time
    most_stuck_state = max(state_visit_counts, key=state_visit_counts.get)

    print("\n--- Experiment Results ---")
    print(f"Success Rate: {success_rate:.2f}%")
    if success_count > 0:
        print(f"Average Length to Resolution (for successful cases): {average_steps:.2f} steps")
    print(f"State where customers get stuck the most: '{most_stuck_state}' (visited {state_visit_counts[most_stuck_state]} times)")
    print("--------------------------\n")


# --- Step 5: Visual Analysis ---
def visualize_chain():
    """
    Creates and displays a visual representation of the Markov chain using NetworkX.
    """
    G = nx.DiGraph()

    # Add nodes (states) to the graph
    for state in STATES:
        G.add_node(state)

    # Add edges with probabilities as labels
    for start_node, transitions in TRANSITION_PROBS.items():
        for end_node, probability in transitions.items():
            # Only draw edges with a non-zero probability to avoid clutter
            if probability > 0:
                G.add_edge(start_node, end_node, weight=probability, label=f"{probability:.2f}")

    # Set up the plot
    pos = nx.spring_layout(G, seed=42)  # Use a seed for reproducible layout
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Markov Chain for Technical Support Diagnosis", size=16)
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # Run the experiments
    run_experiments(num_simulations=1000)

    # Display the visualization
    print("Generating visualization of the Markov chain...")
    visualize_chain()