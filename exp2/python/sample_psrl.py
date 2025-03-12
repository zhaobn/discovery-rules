# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
'''
https://chatgpt.com/share/67cf23f6-4b0c-800c-940d-0ba55dc54407

A simple PSRL example using a finite-horizon MDP with unknown transition probabilities and rewards. 
We'll use Dirichlet-Multinomial and Gaussian-Gamma priors for Bayesian updates.

Task: A Simple Gridworld with Two Actions
Imagine an agent in a one-state MDP with two actions:

- Left (A=0) → Reward sampled from N(u_0, sigma_0^2)
- Right (A=1) → Reward sampled from N(u_1, sigma_1^2)

The transition is trivial (it always stays in the same state),
but the agent does not know the true reward distributions initially. 
It must learn the reward distributions while balancing exploration and exploitation.

Key Steps in PSRL
- Initialize prior beliefs about the rewards (Gaussian-Gamma) and transition model (Dirichlet-Multinomial).
- Sample an MDP from the posterior distribution at the start of each episode.
- Solve the MDP optimally for that sample (i.e., pick the best action greedily).
- Execute the action, observe rewards, and update posterior beliefs.
- Repeat for multiple episodes to learn the optimal policy.

'''

# %%
# Initialize prior parameters (Gaussian-Gamma for reward distributions)
mu_prior = np.array([0.0, 0.0])  # Prior mean for rewards
lambda_prior = np.array([1.0, 1.0])  # Strength of prior (pseudo-counts)
alpha_prior = np.array([1.0, 1.0])  # Shape parameter (prior on variance - inverse gamma)
beta_prior = np.array([1.0, 1.0])  # Scale parameter (prior on variance - inverse gamma)

# %%
# Track observed data
reward_sums = np.zeros(2)  # Sum of rewards per action
reward_counts = np.zeros(2)  # Number of times each action was taken


# %%
num_episodes = 100
num_actions = len(mu_prior)

# Initialize variables to track learning progress
rewards = np.zeros(num_episodes)
means = np.zeros((num_episodes, num_actions)) 
variances = np.zeros((num_episodes, num_actions))

# Run PSRL loop
for episode in range(num_episodes):
    # Step 1: Sample a reward model from posterior (Gaussian-Gamma sampling)
    sampled_variance = 1 / np.random.gamma(alpha_prior + reward_counts, 1 / (beta_prior + 0.5 * reward_sums**2))
    sampled_mean = np.random.normal(
        (lambda_prior * mu_prior + reward_sums) / (lambda_prior + reward_counts),
        np.sqrt(sampled_variance / (lambda_prior + reward_counts))
    )
    
    # Step 2: Solve the sampled MDP (Greedy action selection)
    best_action = np.argmax(sampled_mean)  # Choose action with highest mean reward
    
    # Step 3: Execute action and observe reward (unknown true means)
    true_means = np.array([0.0, 5.0])  # The actual (unknown) rewards
    observed_reward = np.random.normal(true_means[best_action], 1.0)
    
    # Step 4: Update posterior beliefs
    reward_sums[best_action] += observed_reward
    reward_counts[best_action] += 1

    # Store rewards, means, and variances for visualization
    rewards[episode] = observed_reward
    means[episode, :] = sampled_mean
    variances[episode, :] = sampled_variance

    # Print learning progress
    if episode % 10 == 0:
        print(f"Episode {episode}: Best action = {best_action}, Observed reward = {observed_reward:.2f}")

# Print final estimated means
estimated_means = reward_sums / np.maximum(reward_counts, 1)
print("\nFinal estimated means:", estimated_means)

# %%
# Plot learning trajectories
plt.figure(figsize=(12, 6))
for i in range(num_actions):
    plt.plot(np.arange(num_episodes), means[:, i], label=f'Action {i+1} - Mean')

plt.title('Posterior Mean over Time')
plt.xlabel('Time Steps')
plt.ylabel('Mean')
plt.legend(loc='upper right')
plt.show()

# Plot rewards over time
plt.plot(np.arange(num_episodes), rewards, label='Rewards', linestyle='--', color='black', alpha=0.7)

plt.title('Learning Trajectories in Posterior Sampling RL')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.show()

# Plot posterior variances for each action over time
plt.figure(figsize=(12, 6))
for i in range(num_actions):
    plt.plot(np.arange(num_episodes), variances[:, i], label=f'Action {i+1} - Variance')

plt.title('Posterior Variances over Time')
plt.xlabel('Time Steps')
plt.ylabel('Variance')
plt.legend(loc='upper right')
plt.show()
# %%
'''
The code skips skips the division by 2 when updating alpha.

When updating beta, the code simplify the update by 
(1) igoroing the sum of squared errors and using reward_sum^2 instead, and 
(2) ignoring the prior adjusment term.
'''