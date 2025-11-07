from os.path import join, dirname
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

def main():
    # ---------- Experiment configs ----------
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

    # ---------- Loss curves ----------
    plt.figure(figsize=(8, 5))
    for name in results:
        plt.plot(results[name]["history"], label=name.upper())
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Gradient Descent vs RMSprop vs Adam")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(dirname(__file__), 'loss_function_graph.png'))

    # ---------- True vs learned line (best optimizer) ----------

    plt.figure(figsize=(6, 5))
    plt.scatter(X, Y, s=15, alpha=0.6, label="Data")
    plt.plot(X, true_w * X + true_b, linestyle="--", label="True line")
    for name in results.keys():
        name = min(results, key=lambda k: results[k]["history"][-1])
        bw, bb = results[name]["w"], results[name]["b"]        
        plt.plot(X, bw * X + bb, label=f"Learned ({name.upper()})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs Learned Line")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(dirname(__file__), 'true_vs_learned.png'))


if __name__ == "__main__":
    main()
