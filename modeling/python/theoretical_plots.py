# %% 
import numpy as np
import matplotlib.pyplot as plt


# %% 
np.random.seed(0)

generations = 60
runs = 300  # Monte Carlo repetitions

# Cognitive parameters
c = 1.0
d_m = 4
d_triv = 1
alpha = 0.35  # abandonment sensitivity

# Fragment discovery parameters
fragment_size_mean = 1.2   # average depth reduction when successful
fragment_prob_scale = 0.6  # probability scaling for discovering fragment


# True depths for regimes
d_easy = 3      # <= d_m
d_medium = 6    # slightly > d_m
d_hard = 12     # >> d_m


# %% 
def simulate(d_true):
    complexity = np.zeros((runs, generations))
    entropy = np.zeros((runs, generations))

    for r in range(runs):
        d_eff = d_true

        for g in range(generations):

            delta = d_eff - d_m

            if delta > 0:
                epsilon = 1 - np.exp(-alpha * delta)
            else:
                epsilon = 0.0

            # Probability of discovering non-trivial fragment
            p_fragment = (1 - epsilon) * fragment_prob_scale

            if delta <= 0:
                # true rule representable
                d_hat = d_eff
                H = 0.0
            else:
                if np.random.rand() < epsilon:
                    # give up
                    d_hat = d_triv
                    H = 0.1
                else:
                    # approximate boundary depth
                    d_hat = d_m
                    H = 1.0

            complexity[r, g] = c * d_hat
            entropy[r, g] = H

            # Fragment transmission
            if delta > 0 and np.random.rand() < p_fragment:
                reduction = max(1, np.random.poisson(fragment_size_mean))
                d_eff = max(d_m, d_eff - reduction)

    return complexity.mean(axis=0), entropy.mean(axis=0)


# %% 
comp_easy, ent_easy = simulate(d_easy)
comp_med, ent_med = simulate(d_medium)
comp_hard, ent_hard = simulate(d_hard)

# %% 
plt.figure()
plt.plot(comp_easy)
plt.plot(comp_med)
plt.plot(comp_hard)
plt.xlabel("Generation")
plt.ylabel("Linguistic Complexity")
plt.title("Generational Linguistic Complexity")
plt.legend(["Easy", "Medium", "Hard"])
plt.show()

# %% 
plt.figure()
plt.plot(ent_easy)
plt.plot(ent_med)
plt.plot(ent_hard)
plt.xlabel("Generation")
plt.ylabel("Entropy (Diversity)")
plt.title("Generational Diversity")
plt.legend(["Easy", "Medium", "Hard"])
plt.show()


# %%


# -----------------------------
# Parameters
# -----------------------------

c = 1.0
d_m = 4
d_triv = 1
alpha = 0.4
kappa = 0.6

generations = 80
g_vals = np.linspace(0, generations, 800)

# Regime true depths
d_easy = 3
d_medium = 6
d_hard = 12


def compute_curves(d_true):

    Delta0 = d_true - d_m
    D_vals = []

    for g in g_vals:

        # EASY regime
        if Delta0 <= 0:
            D_vals.append(c * d_true)
            continue

        denom = np.exp(alpha * Delta0) - alpha * kappa * g

        # After representability
        if denom <= 1:
            D_vals.append(c * d_true)
            continue

        # Misspecified regime
        epsilon = 1 - 1/denom
        D = (1 - epsilon) * c * d_m + epsilon * c * d_triv

        D_vals.append(D)

    return np.array(D_vals)


# Compute
D_easy = compute_curves(d_easy)
D_med = compute_curves(d_medium)
D_hard = compute_curves(d_hard)


# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(7,4))
plt.plot(g_vals, D_easy)
plt.plot(g_vals, D_med)
plt.plot(g_vals, D_hard)

plt.xlabel("Generation")
plt.ylabel("Linguistic Complexity")
plt.title("Corrected Analytical Generational Complexity")
plt.legend(["Easy", "Medium", "Hard"])
plt.tight_layout()
plt.show()


# %% 
c = 1.0
d_m = 4
d_triv = 1
alpha = 0.35

d_true_values = np.linspace(1, 14, 400)

complexity = []
entropy = []

# -----------------------------
# Closed-form computation
# -----------------------------

for d_true in d_true_values:

    delta = d_true - d_m

    if delta <= 0:
        D = c * d_true
        H = 0.0
    else:
        epsilon = 1 - np.exp(-alpha * delta)
        D = (1 - epsilon) * c * d_m + epsilon * c * d_triv
        
        # Analytical entropy of 2-outcome mixture
        if epsilon == 0 or epsilon == 1:
            H = 0.0
        else:
            H = - (epsilon * np.log(epsilon) +
                   (1 - epsilon) * np.log(1 - epsilon))

    complexity.append(D)
    entropy.append(H)

complexity = np.array(complexity)
entropy = np.array(entropy)


plt.figure(figsize=(6,4))
plt.plot(d_true_values, complexity, linewidth=2)
plt.axvline(d_m, linestyle="--")
plt.xlabel("True Rule Depth")
plt.ylabel("Expected Linguistic Complexity")
plt.title("Analytical Inverted-U (Single Generation)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.plot(d_true_values, entropy, linewidth=2)
plt.axvline(d_m, linestyle="--")
plt.xlabel("True Rule Depth")
plt.ylabel("Entropy")
plt.title("Diversity Across Difficulty")
plt.tight_layout()
plt.show()



