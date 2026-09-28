# DAPO implementation

This project is an educational implementation of the reinforcement-learning
workflow behind DAPO (**Decoupled Clip and Dynamic sAmpling Policy
Optimization**). DAPO extends GRPO-style training to improve mathematical
reasoning while avoiding a learned reward model and critic/value network.

The project starts from Qwen2.5-3B-Instruct and trains it on verifiable
Countdown arithmetic problems. For each training step it:

1. samples eight candidate answers for each of 32 questions;
2. scores answer correctness and required output format with deterministic
   rules;
3. normalizes rewards within each question's response group to obtain
   group-relative advantages;
4. applies the policy-gradient loss only to generated response tokens;
5. accumulates microbatch gradients, clips the gradient norm, updates the
   policy, evaluates periodically, and records TensorBoard metrics.

```text
Qwen2.5-3B-Instruct
    -> grouped response rollouts
    -> verifiable rewards
    -> group-relative advantages
    -> response-token policy update
    -> improved Countdown reasoning policy
```

The main components are:

- `train.py`: training loop, evaluation, logging, and checkpoints;
- `grpo.py`: grouped rollout generation, reward normalization, and policy
  update;
- `countdown_task.py`: dataset prompts and rule-based rewards;
- `qwen2_model.py`: local Qwen2 transformer and KV-cache implementation;
- `submit-DAPO-qwen253b.sh`: Slurm launcher for an A100 GPU.

This code demonstrates the foundation of DAPO-style training, but it is not a
complete reproduction of the published DAPO system. In particular, the current
policy update is closer to a basic group-relative REINFORCE/GRPO-style update:
it does not yet store old-policy probabilities or implement DAPO's asymmetric
Clip-Higher objective, dynamic sampling that filters zero-gradient groups, or
soft overlong-reward shaping. Production DAPO also requires distributed
rollout/training infrastructure, larger datasets, and substantially more
compute.

Run it on the server with:

```bash
cd ~/scratch/dips_project/reinforcement_learning/agent_rl_learning/day07_GRPO/dapo
sbatch submit-DAPO-qwen253b.sh
```
