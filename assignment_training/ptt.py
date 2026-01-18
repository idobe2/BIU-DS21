import pandas as pd
import random
from datasets import load_dataset
import re

# 1. Load the dataset
ds = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")  # Sample of 1000 examples
df = pd.DataFrame(ds)

# 2. Add user_type randomly to each sample
user_types = ["strict", "friendly", "efficient"]
df["user_type"] = [random.choice(user_types) for _ in range(len(df))]


# 3. Define a fictitious reward function (based on the presentation - answer length)
def fictitious_reward_v0(text):
    """Simple reward based on word count."""
    return len(text.split())


# 4. Build the basic policy (selects answer with higher reward)
def policy_v0(row):
    """Basic policy that chooses response with higher word count."""
    reward_a = fictitious_reward_v0(row["chosen"])  # Assume response A is 'chosen'
    reward_b = fictitious_reward_v0(row["rejected"])  # Assume response B is 'rejected'

    return "A" if reward_a >= reward_b else "B"


# Run the policy
df["prediction_v0"] = df.apply(policy_v0, axis=1)

df["is_correct"] = (df["prediction_v0"] == "A").astype(int)

# Create measurement table (Accuracy for each user type)
accuracy_table = df.groupby("user_type")["is_correct"].mean().reset_index()
accuracy_table.columns = ["User Type", "Accuracy"]

# Calculate overall accuracy
overall_accuracy = df["is_correct"].mean()

print("--- Policy v0: Performance Table ---")
print(accuracy_table)
print(f"\nGeneral Accuracy: {overall_accuracy:.2f}")


def extract_features(text):
    """Extract features from text: length, politeness, and structure."""
    # 1. Length (normalized - divided by 100 for smaller range)
    length = len(text.split()) / 100

    # 2. Politeness (search for keyword phrases)
    polite_words = ["please", "thank", "sorry", "apologize", "happy to help", "welcome"]
    polite_count = sum(1 for word in polite_words if word in text.lower())

    # 3. Structure (does it have lists? Sometimes indicates efficiency/order)
    has_list = 1 if bool(re.search(r"\d\.|-", text)) else 0

    return [length, polite_count, has_list]


weights_map = {
    "friendly": [0.5, 2.0, 0.2],  # Strongly prefers politeness (2.0) and length (0.5)
    "efficient": [-1.5, 0.0, 1.0],  # Dislikes length (-1.5), prefers lists/order (1.0)
    "strict": [0.1, -1.0, 0.5],  # Dislikes excessive politeness (-1.0), wants conciseness
}


def context_aware_reward(text, user_type):
    """Calculate reward score based on user type preferences."""
    features = extract_features(text)
    weights = weights_map[user_type]

    # Calculate weighted sum: (f1*w1) + (f2*w2) + (f3*w3)
    reward = sum(f * w for f, w in zip(features, weights))
    return reward


def policy_v1(row):
    """Context-aware policy that considers user type preferences."""
    r_a = context_aware_reward(row["chosen"], row["user_type"])
    r_b = context_aware_reward(row["rejected"], row["user_type"])
    return "A" if r_a >= r_b else "B"


# Run v1 policy and calculate new accuracy
df["prediction_v1"] = df.apply(policy_v1, axis=1)
df["is_correct_v1"] = (df["prediction_v1"] == "A").astype(int)
new_accuracy = df.groupby("user_type")["is_correct_v1"].mean()

# Calculate improvement percentage
comparison = pd.DataFrame(
    {
        "User Type": accuracy_table["User Type"],
        "Accuracy v0 (Baseline)": accuracy_table["Accuracy"],
        "Accuracy v1 (Context-Aware)": new_accuracy.values,
    }
)

comparison["Improvement (%)"] = (
    comparison["Accuracy v1 (Context-Aware)"] - comparison["Accuracy v0 (Baseline)"]
) * 100

print("\n--- Performance Table: v0 vs v1 ---")
print(comparison)
print(f"\nGeneral Accuracy v1: {df['is_correct_v1'].mean():.2f}")

# Define two responses for comparison for Friendly user
hacked_text = "Please please thank you thank you sorry sorry happy to help please thank you."
good_text = "The capital of France is Paris. I hope this helps you!"

# Calculate the Reward (what the model thinks)
reward_hacked = context_aware_reward(hacked_text, 'friendly')
reward_good = context_aware_reward(good_text, 'friendly')

# Define Human Quality (subjective score on 0-1 scale)
human_quality_hacked = 0.1  # Very poor, gibberish
human_quality_good = 0.9    # Excellent, correct and helpful response

print(f"{'Metric':<20} | {'Good Response':<15} | {'Hacked Response':<15}")
print("-" * 55)
print(f"{'Reward Score (v1)':<20} | {reward_good:<15.2f} | {reward_hacked:<15.2f}")
print(f"{'Human Quality':<20} | {human_quality_good:<15.2f} | {human_quality_hacked:<15.2f}")