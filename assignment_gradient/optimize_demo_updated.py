import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# ---------- Synthetic data ----------
N = 200
X = np.linspace(-3, 3, N)
true_w, true_b = 3.0, 2.0
noise = np.random.normal(0, 1.0, size=N)
Y = true_w * X + true_b + noise

# ---------- Loss and gradient ----------
def mse(w, b, X, Y):
    pred = w * X + b
    return np.mean((pred - Y) ** 2)

def grad(w, b, X, Y):
    pred = w * X + b
    err = pred - Y
    dw = (2.0 / len(X)) * np.sum(err * X)
    db = (2.0 / len(X)) * np.sum(err)
    return dw, db

# ---------- Optimizers ----------
def gd_step(w, b, dw, db, lr):
    # Basic Gradient Descent
    w -= lr * dw
    b -= lr * db
    return w, b

def rmsprop_step(w, b, dw, db, s_w, s_b, lr, beta=0.9, eps=1e-8):
    # RMSprop: exponential moving average of squared gradients
    s_w = beta * s_w + (1 - beta) * (dw ** 2)
    s_b = beta * s_b + (1 - beta) * (db ** 2)

    w -= (lr / (np.sqrt(s_w) + eps)) * dw
    b -= (lr / (np.sqrt(s_b) + eps)) * db
    return w, b, s_w, s_b

def adam_step(w, b, dw, db, m_w, v_w, m_b, v_b,
              lr, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
    # Adam: first (m) and second (v) moment estimates + bias correction
    m_w = beta1 * m_w + (1 - beta1) * dw
    v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
    m_b = beta1 * m_b + (1 - beta1) * db
    v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

    # Bias correction
    m_w_hat = m_w / (1 - beta1 ** t)
    v_w_hat = v_w / (1 - beta2 ** t)
    m_b_hat = m_b / (1 - beta1 ** t)
    v_b_hat = v_b / (1 - beta2 ** t)

    w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
    b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    return w, b, m_w, v_w, m_b, v_b

# ---------- Training loop ----------
def train(optimizer_name="gd", epochs=200, lr=0.05):
    w, b = np.random.randn(), np.random.randn()
    history = []

    # Internal states
    s_w = s_b = 0.0                 # RMSprop
    m_w = v_w = m_b = v_b = 0.0     # Adam

    for t in range(1, epochs + 1):
        dw, db = grad(w, b, X, Y)

        if optimizer_name == "gd":
            w, b = gd_step(w, b, dw, db, lr)

        elif optimizer_name == "rmsprop":
            w, b, s_w, s_b = rmsprop_step(w, b, dw, db, s_w, s_b, lr, beta=0.9)

        elif optimizer_name == "adam":
            w, b, m_w, v_w, m_b, v_b = adam_step(
                w, b, dw, db, m_w, v_w, m_b, v_b, lr,
                beta1=0.9, beta2=0.999, eps=1e-8, t=t
            )
        else:
            raise ValueError("Unknown optimizer. Use: 'gd', 'rmsprop', or 'adam'.")

        loss = mse(w, b, X, Y)
        history.append(loss)

        # Basic sanity check
        if np.isnan(loss) or np.isinf(loss):
            raise ValueError("Loss exploded. Try lowering the learning rate.")

    return w, b, np.array(history)

# ---------- MINIMAL ADDITIONS FOR Q8–Q10 ----------
def run_suite(lr=0.01, epochs=300, label="run"):  # CHANGE: helper to run all 3 with given lr/epochs
    results = {}
    for name in ("gd", "rmsprop", "adam"):
        w, b, hist = train(optimizer_name=name, epochs=epochs, lr=lr)
        results[name] = {"w": w, "b": b, "history": hist}
    print(f"\n=== {label} (lr={lr}, epochs={epochs}) ===")  # CHANGE: console summary
    for name in ("gd", "rmsprop", "adam"):
        r = results[name]
        print(f"{name.upper():7s} -> w={r['w']:.3f}, b={r['b']:.3f}, final MSE={r['history'][-1]:.4f}")
    return results

def set_noise(std=1.0, seed=42):  # CHANGE: re-generate Y for Q9 without touching rest of code
    global Y, noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, size=N)
    Y = true_w * X + true_b + noise
# -----------------------------------

def main():
    # ---------- Experiment configs (original behavior kept) ----------
    configs = [
        ("gd", 0.01),
        ("rmsprop", 0.01),
        ("adam", 0.01),
    ]

    results = {}
    for name, lr in configs:
        w, b, hist = train(optimizer_name=name, epochs=300, lr=lr)
        results[name] = {"w": w, "b": b, "history": hist}
        print(f"{name.upper():7s} -> w={w:.3f}, b={b:.3f}, final MSE={hist[-1]:.4f}")

    # ---------- Loss curves (original) ----------
    plt.figure(figsize=(8, 5))
    for name in results:
        plt.plot(results[name]["history"], label=name.upper())
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Gradient Descent vs RMSprop vs Adam")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- True vs learned line (original) ----------
    best_name = min(results, key=lambda k: results[k]["history"][-1])
    bw, bb = results[best_name]["w"], results[best_name]["b"]

    plt.figure(figsize=(6, 5))
    plt.scatter(X, Y, s=15, alpha=0.6, label="Data")
    plt.plot(X, true_w * X + true_b, linestyle="--", label="True line")
    plt.plot(X, bw * X + bb, label=f"Learned ({best_name.upper()})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs Learned Line")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- Q8: LR sweep (added) ----------
    run_suite(lr=0.01, epochs=300, label="Q8 baseline")      # CHANGE
    run_suite(lr=0.10, epochs=300, label="Q8 high lr=0.10")   # CHANGE
    run_suite(lr=0.001, epochs=300, label="Q8 low lr=0.001")  # CHANGE

    # ---------- Q9: add noise (std=3.0) ----------
    set_noise(std=3.0)                                        # CHANGE
    run_suite(lr=0.01, epochs=300, label="Q9 noisy std=3.0")  # CHANGE
    set_noise(std=1.0)                                        # CHANGE reset

    # ---------- Q10: extend epochs to 1000 ----------
    run_suite(lr=0.01, epochs=1000, label="Q10 epochs=1000")  # CHANGE

if __name__ == "__main__":
    main()
