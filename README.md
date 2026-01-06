## GRPO (Group Relative Policy Optimization)

### What is it ?
GRPO is a reainforcement learning algorithm for LLM alignment introduced in the by DeepSeek in their DeepSeek-math (2024) and DeepSee-R1 papers. It is variant of PPO(Proximal Policy Optimization) that elimenates the need for a separate critic/value network computing advantages from group of sampled responses. 

The core inotvation: instead of traning a value function to estimate baseline (as in standard PPO), GRPO generates multiple responses per prompt and uses their relative rewards within the group as the baseline. 

## How it works ? 

Given a prompt $x$, the current policy $\pi_\theta$ generates a group of $K$ responses: $(y_1, y_2, \ldots, y_K)$. Each response is scored by a reward model:

$$r_i = r(x, y_i)$$

### Step 1: Compute Group-Relative Advantages

Instead of using a learned value function, GRPO normalizes rewards within each group:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_K)}{\text{std}(r_1, \ldots, r_K)}$$

Or more explicitly:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r}, \quad \text{where} \quad \bar{r} = \frac{1}{K}\sum_{j=1}^{K} r_j, \quad \sigma_r = \sqrt{\frac{1}{K}\sum_{j=1}^{K}(r_j - \bar{r})^2}$$

### Step 2: GRPO Objective with Clipping

GRPO inherits PPO's clipped surrogate objective to ensure stable updates:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \, y_i \sim \pi_{\text{old}}(\cdot|x)} \left[ \frac{1}{K} \sum_{i=1}^{K} \min\left( \rho_i A_i, \, \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) A_i \right) \right]$$

Where the importance sampling ratio is:

$$\rho_i = \frac{\pi_\theta(y_i | x)}{\pi_{\text{old}}(y_i | x)}$$

### Step 3: KL Regularization

To prevent the policy from drifting too far from a reference model $\pi_{\text{ref}}$ (typically the SFT model), a KL penalty is added:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x, y_i} \left[ \frac{1}{K} \sum_{i=1}^{K} \min\left( \rho_i A_i, \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) A_i \right) - \beta \cdot D_{\text{KL}}\left( \pi_\theta \parallel \pi_{\text{ref}} \right) \right]$$

In practice, the KL divergence is often estimated per-token using:

$$D_{\text{KL}} \approx \frac{1}{T} \sum_{t=1}^{T} \left( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right)$$

where $T$ is the sequence length.
