from numpy import exp, clip, mean, std

class GRPO:
    """
    Group Relative Policy Optimization (GRPO) algorithm implementation.
    """
    def __init__(self, policy, ref_policy, reward_model, K=4, epsilon=0.2, beta=0.01):
        """
        Initialize GRPO with given parameters.
        
        Args:
            policy: Current policy π_θ
            ref_policy: Reference policy π_ref (frozen SFT model)
            reward_model: Reward function r(x, y)
            K: Number of responses per prompt
            epsilon: PPO clipping parameter
            beta: KL penalty coefficient
        """
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.K = K
        self.epsilon = epsilon
        self.beta = beta
        self.all_losses = []

    def grpo_step(self, prompts):
        """
        One GRPO training step.
        
        Args:
            prompts: Batch of prompts
            policy: Current policy π_θ
            ref_policy: Reference policy π_ref (frozen SFT model)
            reward_model: Reward function r(x, y)
            K: Number of responses per prompt
            epsilon: PPO clipping parameter
            beta: KL penalty coefficient
        """
        all_losses = []

        for x in prompts:
            # Generate K responses from old policy
            responses = [self.policy.generate(x) for _ in range(self.K)]

            # Get reward for each response
            rewards = [self.reward_model(x, y) for y in responses]

            # Compute group-relative advantages
            mean_reward = mean(rewards)
            std_r = std(rewards) + 1e-8  # Avoid division by zero
            advantages = [(r - mean_reward) / std_r for r in rewards]

            for y_i, A_i in zip(responses, advantages):
                # Importance sampling ratio
                log_prob_new = self.policy.log_prob(x, y_i)
                log_prob_old = self.ref_policy.log_prob(x, y_i)
                ratio = exp(log_prob_new - log_prob_old)

                # Clipped surrogate objective
                surr1 = ratio * A_i
                surr2 = clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_i
                policy_loss = -min(surr1, surr2)

                # KL penalty
                log_prob_ref = self.ref_policy.log_prob(x, y_i)
                kl = (log_prob_new - log_prob_ref).mean()

                loss = policy_loss + self.beta * kl

                all_losses.append(loss)
            
            return mean(all_losses)

        
    def step(self, prompts):
        """
        One GRPO training step.
        """
        loss = self.grpo_step(prompts, self.policy, self.ref_policy, self.reward_model, self.K, self.epsilon, self.beta)
        self.all_losses.append(loss)
        return loss
    
    def __call__(self, prompts):
        return self.step(prompts)
from numpy import exp, clip, mean, std

class GRPO:
    """
    Group Relative Policy Optimization (GRPO) algorithm implementation.
    """
    def __init__(self, policy, ref_policy, reward_model, K=4, epsilon=0.2, beta=0.01):
        """
        Initialize GRPO with given parameters.
        
        Args:
            policy: Current policy π_θ
            ref_policy: Reference policy π_ref (frozen SFT model)
            reward_model: Reward function r(x, y)
            K: Number of responses per prompt
            epsilon: PPO clipping parameter
            beta: KL penalty coefficient
        """
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.K = K
        self.epsilon = epsilon
        self.beta = beta
        self.all_losses = []

    def grpo_step(self, prompts):
        """
        One GRPO training step.
        
        Args:
            prompts: Batch of prompts
            policy: Current policy π_θ
            ref_policy: Reference policy π_ref (frozen SFT model)
            reward_model: Reward function r(x, y)
            K: Number of responses per prompt
            epsilon: PPO clipping parameter
            beta: KL penalty coefficient
        """
        all_losses = []

        for x in prompts:
            # Generate K responses from old policy
            responses = [self.policy.generate(x) for _ in range(self.K)]

            # Get reward for each response
            rewards = [self.reward_model(x, y) for y in responses]

            # Compute group-relative advantages
            mean_reward = mean(rewards)
            std_r = std(rewards) + 1e-8  # Avoid division by zero
            advantages = [(r - mean_reward) / std_r for r in rewards]

            for y_i, A_i in zip(responses, advantages):
                # Importance sampling ratio
                log_prob_new = self.policy.log_prob(x, y_i)
                log_prob_old = self.ref_policy.log_prob(x, y_i)
                ratio = exp(log_prob_new - log_prob_old)

                # Clipped surrogate objective
                surr1 = ratio * A_i
                surr2 = clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_i
                policy_loss = -min(surr1, surr2)

                # KL penalty
                log_prob_ref = self.ref_policy.log_prob(x, y_i)
                kl = (log_prob_new - log_prob_ref).mean()

                loss = policy_loss + self.beta * kl

                all_losses.append(loss)
            
            return mean(all_losses)

        
    def step(self, prompts):
        """
        One GRPO training step.
        """
        loss = self.grpo_step(prompts, self.policy, self.ref_policy, self.reward_model, self.K, self.epsilon, self.beta)
        self.all_losses.append(loss)
        return loss
    
    def __call__(self, prompts):
        return self.step(prompts)
