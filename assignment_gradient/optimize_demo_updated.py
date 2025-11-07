import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Synthetic data
N = 200
X = np.linspace(-3, 3, N)
true_w, true_b = 3.0, 2.0
noise = np.random.normal(0, 1.0, size=N)
Y = true_w * X + true_b + noise

# Loss and gradient
def mse(w, b, X, Y):
    pred = w * X + b
    return np.mean((pred - Y) ** 2)

def grad(w, b, X, Y):
    pred = w * X + b
    err = pred - Y
    dw = (2.0 / len(X)) * np.sum(err * X)
    db = (2.0 / len(X)) * np.sum(err)
    return dw, db

# Optimizers
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

# Training loop
def train(optimizer_name="gd", epochs=200, lr=0.05,
          lr_w=None, lr_b=None):
    w, b = np.random.randn(), np.random.randn()
    history = []

    # Internal states
    s_w = s_b = 0.0                 # RMSprop
    m_w = v_w = m_b = v_b = 0.0     # Adam

    # Q2: scale gradients if separate lr_w/lr_b were provided
    scale_w = (lr_w / lr) if (lr_w is not None) else 1.0
    scale_b = (lr_b / lr) if (lr_b is not None) else 1.0

    for t in range(1, epochs + 1):
        dw, db = grad(w, b, X, Y)
        dw_use = dw * scale_w
        db_use = db * scale_b 

        try:
            if optimizer_name == "gd":
                w, b = gd_step(w, b, dw_use, db_use, lr)

            elif optimizer_name == "rmsprop":
                w, b, s_w, s_b = rmsprop_step(w, b, dw_use, db_use, s_w, s_b, lr, beta=0.9)

            elif optimizer_name == "adam":
                w, b, m_w, v_w, m_b, v_b = adam_step(
                    w, b, dw_use, db_use, m_w, v_w, m_b, v_b, lr,
                    beta1=0.9, beta2=0.999, eps=1e-8, t=t
                )
            else:
                raise ValueError("Unknown optimizer. Use: 'gd', 'rmsprop', or 'adam'.")

            loss = mse(w, b, X, Y)
            history.append(loss)

            # Basic sanity check
            if np.isnan(loss) or np.isinf(loss):
                raise FloatingPointError("Loss exploded")

        except FloatingPointError:
            # Q3: mark divergence gracefully (so we can compare who fails first)
            history.append(np.inf)
            break

    return w, b, np.array(history)

# Q8–Q10
def run_suite(lr=0.01, epochs=300, label="run"):
    results = {}
    for name in ("gd", "rmsprop", "adam"):
        w, b, hist = train(optimizer_name=name, epochs=epochs, lr=lr)
        results[name] = {"w": w, "b": b, "history": hist}
    print(f"\n----- {label} (lr={lr}, epochs={epochs}) -----")
    for name in ("gd", "rmsprop", "adam"):
        r = results[name]
        final = r['history'][-1]
        tag = "DIVERGED" if not np.isfinite(final) else f"{final:.4f}"
        print(f"{name.upper():7s} -> w={r['w']:.3f}, b={r['b']:.3f}, final MSE={tag}")
    return results

def set_noise(std=1.0, seed=42):
    # re-generate Y for noise experiments
    global Y, noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, size=N)
    Y = true_w * X + true_b + noise

# Q1–Q6
def run_suite_safe(lr=0.01, epochs=300, label="run"):
    try:
        return run_suite(lr=lr, epochs=epochs, label=label)
    except Exception as e:
        print(f"[{label}] run failed: {e}")

def run_suite_per_param(lr=0.01, lr_w=None, lr_b=None, epochs=300, label="per-param"):
    results = {}
    for name in ("gd", "rmsprop", "adam"):
        w, b, hist = train(optimizer_name=name, epochs=epochs, lr=lr, lr_w=lr_w, lr_b=lr_b)
        results[name] = {"w": w, "b": b, "history": hist}
    print(f"\n----- {label} (base lr={lr}, lr_w={lr_w}, lr_b={lr_b}, epochs={epochs}) -----")
    for name in ("gd", "rmsprop", "adam"):
        r = results[name]
        final = r['history'][-1]
        tag = "DIVERGED" if not np.isfinite(final) else f"{final:.4f}"
        print(f"{name.upper():7s} -> w={r['w']:.3f}, b={r['b']:.3f}, final MSE={tag}")
    return results

# Q7–Q9
def epochs_to_threshold(history, thr=1.0):
    """Return first epoch index (1-based) where loss < thr, or None if never."""
    for i, v in enumerate(history, start=1):
        if np.isfinite(v) and v < thr:
            return i
    return None

def set_dataset(N_new=None, x_min=None, x_max=None, noise_std=None, seed=42):
    """Regenerate (X,Y) with optional new N / range / noise; keeps others as-is if None."""
    global N, X, Y, noise
    if N_new is None:
        N_new = N
    if x_min is None or x_max is None:
        x_min, x_max = X.min(), X.max()
    if noise_std is None:
        # keep current noise std
        noise_std = noise.std() if np.isfinite(noise).all() else 1.0
    rng = np.random.default_rng(seed)
    N = int(N_new)
    X = np.linspace(x_min, x_max, N)
    noise = rng.normal(0, noise_std, size=N)
    Y = true_w * X + true_b + noise


def main():
    # Experiment configs (original behavior kept)
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

    # Loss curves (original)
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

    # True vs learned line (original)
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

    # Q8: LR sweep (existing)
    run_suite(lr=0.01, epochs=300, label="Q8 baseline")
    run_suite(lr=0.10, epochs=300, label="Q8 high lr=0.10")
    run_suite(lr=0.001, epochs=300, label="Q8 low lr=0.001")

    # Q9: add noise (existing)
    set_noise(std=3.0)
    run_suite(lr=0.01, epochs=300, label="Q9 noisy std=3.0")
    set_noise(std=1.0)  # reset

    # Q10: extend epochs (existing)
    run_suite(lr=0.01, epochs=1000, label="Q10 epochs=1000")

    # Q1: very small lr -> slow/“stuck”
    run_suite_safe(lr=0.0001, epochs=300, label="Q1 lr=1e-4")

    # Q2: different lr for w and b (e.g., w slower, b faster)
    run_suite_per_param(lr=0.01, lr_w=0.01, lr_b=0.10, epochs=300, label="Q2 per-parameter LRs")

    # Q3: very high lr -> who diverges first?
    run_suite_safe(lr=1.0, epochs=300, label="Q3 high lr=1.0")

    # Q4: increase sample size N=1000 -> smoother/steadier loss
    set_dataset(N_new=1000)
    run_suite_safe(lr=0.01, epochs=300, label="Q4 N=1000")
    # restore defaults
    set_dataset(N_new=200)

    # Q5: tiny sample size N=20 -> noisier, Adam often best
    set_dataset(N_new=20)
    run_suite_safe(lr=0.01, epochs=300, label="Q5 N=20")
    set_dataset(N_new=200)

    # Q6: widen X range to [-100, 100] -> need smaller lr, otherwise unstable
    set_dataset(N_new=200, x_min=-100, x_max=100)
    run_suite_safe(lr=0.001, epochs=300, label="Q6 X in [-100,100], lr=0.001")  # safer lr
    # restore defaults
    set_dataset(N_new=200, x_min=-3, x_max=3)

    # Q7: How many epochs until Loss < 1? 
    print("\n----- Q7: epochs until Loss < 1 (lr=0.01, epochs=1000) -----")
    q7_results = {}
    for name in ("gd", "rmsprop", "adam"):
        w, b, hist = train(optimizer_name=name, epochs=1000, lr=0.01)
        e = epochs_to_threshold(hist, thr=1.0)
        q7_results[name] = e
        print(f"{name.upper():7s} -> epochs_to_<1: {e if e is not None else 'NEVER'}")

    # Q8: Wall-clock time for 1000 epochs
    import time
    print("\n----- Q8: wall-clock time for 1000 epochs (lr=0.01) -----")
    q8_times = {}
    for name in ("gd", "rmsprop", "adam"):
        t0 = time.time()
        _ = train(optimizer_name=name, epochs=1000, lr=0.01)
        t1 = time.time()
        q8_times[name] = t1 - t0
        print(f"{name.upper():7s} -> {q8_times[name]:.4f} sec")

    # Q9: Who is closest to the true parameters (w=3, b=2)?
    print("\n----- Q9: distance to true params (w=3, b=2) after 1000 epochs -----")
    q9_dist = {}
    for name in ("gd", "rmsprop", "adam"):
        w, b, hist = train(optimizer_name=name, epochs=1000, lr=0.01)
        dist = np.sqrt((w - true_w)**2 + (b - true_b)**2)
        q9_dist[name] = dist
        print(f"{name.upper():7s} -> w={w:.3f}, b={b:.3f}, dist={dist:.4f}")

if __name__ == "__main__":
    main()
