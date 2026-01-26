# **For empirical research (for myself only).**

# Usage in run.ai clusters (A100 & L40S) and B200.

Large Storage: `/shared_data0/hnwong`
Check quota: `quota -vs`

## run.ai
- Access to cluster `ssh hnwong@locust-login.seas.upenn.edu`
- `runai login` username `hnwong@upenn.edu`
- Delete job: runai `delete job honam`
- Submit interactive job:
```sh
runai submit honam \
   -i hnwong2025/base:latest \
   --attach \
   --interactive \
   --tty \
   --stdin \
   -v /home/hnwong:/home/hnwong \
   -v /shared_data0:/shared_data0 \
   --cpu 8 \
   -g 1 \
   --large-shm \
   --memory 128G \
   --working-dir /home/hnwong \
   -e HOME=/home/hnwong \
   --service-type=nodeport --port 30025:22 \
   -- /usr/sbin/sshd -D # For running another job, you need to change the port number 30025
```
and `runai port-forward honam --port 30025:30025` (optional?)

- Jupyter notebook: Create using ui interface in run.ai
- Access Tensorboard: `runai port-forward honam --port 6006:6006` (Forward login node's port to job's port), `ssh -L 6006:localhost:6006 hnwong@locust-login.seas.upenn.edu` (connect local machine's 6006 port to the login node)

## Access Wharton Stats Cluster (L40S)
We still use the locust login node. Then we launch
```sh
#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --time=00:05:00 ## time limit of 5 minutes
#SBATCH --gres=gpu:1 ## request 1 GPU.  You can also request specific types
#SBATCH --ntasks=1 ## instances of the program to run, typically 1.  
```

## B200 nodes (2.5x time faster than A100)
```sh
kinit hnwong@UPENN.EDU
ssh hnwong@login.betty.parcc.upenn.edu
```
- home directory: `/vast/home/h/hnwong`
- find own jobs: `squeue -u $USER`
- Cancel the job: `scancel <job_id>`
- run job: `srun --partition=dgx-b200   --pty --container-image=hnwong2025/base:latest   bash`
- more complicated one, need to find how to maps home addresses correctly:
```sh
srun --export=NONE \
  --partition=dgx-b200 \
  --container-image=docker://hnwong2025/base:latest \
  --container-mounts=/vast/home/h/hnwong:/home/hnwong \
  --container-workdir=/home/hnwong \
  --cpus-per-task=8 --gpus=1 --mem=128G --pty --time=01:00:00 \
  bash -lc '
    P="$PATH";
    env -i \
      PATH="$P" \
      TERM=xterm-256color \
      HOME=/home/hnwong \
      XDG_CACHE_HOME=/home/hnwong/.cache \
      UV_CACHE_DIR=/home/hnwong/.cache/uv \
      PS1="hnwong@\\h:\\w\\$ " \
      bash
  '
```

## Transmit files from run.ai node to B200 node
```
scp hnwong@locust-login.seas.upenn.edu:/shared_data0/hnwong/cache/tinystories_train_maxlen_512.npy \
    hnwong@login.betty.parcc.upenn.edu:~/.cache 
```

## Docker commands:
```sh
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t hnwong2025/base:latest base
docker push hnwong2025/base:latest
```

# Git command:
## Have the wrong commit
(for example: adding large file) and want to modify it before pushing to remote:

- Keeps a copy of your current state in case anything goes wrong:
`git branch backup-before-rewrite`
- Prevents untracked local files from blocking rebase checkouts. `git stash -u -m "temp stash for rebase`
- Rebase to previous commit `git rebase -i HEAD~2` (# of steps depends on how many commits you have made)
- Change the line of the wrong commit to `edit`, and remove unnecessary commits afterwards
- At that commit, do things you want to, to fix the wrong commit. For example: for wrong large file adding `git rm --cached path/to/large_file`.
- Check this by `git rev-list --objects --all | grep 'path/to/large_file' || echo "✅ Large file removed from history`.
- Finally, `git push origin <branch>`.

`git stash`: temporarily save changes in your working directory and index (staged changes) that are not yet ready to be committed. 
Push to private repository `git remote set-url origin https://Matheart:<api_key>@github.com/Matheart/<project>.git`

## Fork some remote repository, and clone the fork locally, want to update the local repository with the new commits in remote repository
### inside your local clone
git remote add upstream https://github.com/original/repo.git   # only once
git fetch upstream

### update your main branch
git checkout main
git merge upstream/main          # or: git rebase upstream/main
# then update your fork on GitHub if desired
git push origin main

# Run Jupyter notebook inside server
```sh
uv pip install ipykernel
uv run python -m ipykernel install --user --name=<project-name> --display-name="Python <project-name>"
```

# Hosted platform 
https://modal.com/

Command
```py
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .pip_install("torch")
    .add_local_python_source("tensor_initializations", "optimization_algorithms", "synthetic_data", "utils", "models", "looper")
)
@app.function(image=training_image, gpu="A100-40GB", timeout=3600)
def get_results(params):
    return looper(params)
@app.local_entrypoint()
def main():
    inputs = [
        run_parameters
    ]
    for result in get_results.map(inputs):
        save_results(result)
```


# Deep Learning Experiments

Choose GPU `export CUDA_VISIBLE_DEVICES=1`

## Learning Rate Scaling Rule
Square root scaling rule, maximizing batch size as much as possible, and scaling lr as sqrt(scaled factor of batch size).
The learning rate scales inversely with model's dimension.

## Torch profiling
### Timing
Most simplest way, 
```py
                torch.cuda.synchronize(); sync_start = time.time()
                loss = loss.item()
                torch.cuda.synchronize(); sync_time = time.time() - sync_start
```
Ignore the first time counting since it may take long (pre-loading etc.), and need to take average over multiple runs.

```py
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/shared_data0/hnwong/logs/profile'),
    with_stack=True
) as prof:
    for step in range(steps):
        train_step() 
        prof.step()
```

`uv add torch_tb_profiler`, then `uv run tensorboard --logdir=/shared_data0/hnwong/logs/profile --port=6006 --bind_all --load_fast=false`. After that we need to handle port forwarding operations. `runai port-forward honam --port 6006:6006` (Forward login node's port to job's port), `ssh -L 6006:localhost:6006 hnwong@locust-login.seas.upenn.edu` (connect local machine's 6006 port to the login node)

## Parrallelize many small experiments in a single GPU

We use `small_exp` as a proxy to small experiments in practice, and we test and compare two approaches, sequential execution and parallel execution.

When experiments inside one batch take approximate amount of time, parallel execution saves some time e.g. `num_of_workers = 4`, but would introduce overhead if `num_of_workers` gets too large. When they have different amount of time (i.e. when varying `batch size` or `width`), parallel execution might be worse than sequential execution. To verify the conclusion we draw above, can run and compare time for code inside `small_exp` folder.

This tells us when considering parallelism:
- Be sure for every batch of experiments, **their execution time should be almost the same**, i.e. should not run experiments of `width = 64, 128` in parallel.
- Use **`num_of_workers = 4`** and **should not set it as too large to avoid overhead**.
- The time can be reduced but not a lot.

Refer to `small_exp/small_exp_multiprocessing.py` for template.

`jax` is actually more powerful when dealing with a large amount of small synthetic experiments.


## Distributed training
TO-DO

## Accelerate Transformer training time on a single GPU
Rememeber don't have frequent CPU-GPU communication during batches (e.g. `.item()`, `.to(device)`)
### Change to other precision
#### TF32 (Enable Tensor Core)
```py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

#### Mix-precision training
Enable `bf16` this can have massive speedup, but need to be careful on where to enable `bf16`.

### `torch.compile()`
Almost one-line, `model = torch.compile(model)`, it captures model’s forward/backward pass once, fuses and optimizes operations, and generate efficient GPU kernels. This always leads to 1.5-2x speedup.

### Increase batch size
Increasing batch size can sometimes be more efficient. Larger batch can give a more accurate estimate of the gradient, can have further decrease of loss after a large number of steps, compared with small batch sizes.

## Hyperparameter sweep experiments using wandb

## jax
https://docs.jax.dev/en/latest/index.html
- Guideline: JAX internally uses functional programming model. So all the funtions should be pure (No side effect i.e. `print` inside function, or using external variables). Don't use iterator or might have errors / unexpected result. For debug printing, use `jax.debug.print()`.
- `jax.jit`, `jax.map`, `jax.grad` are often applicable to static shapes only, but the scenarios that need dynamic shapes can always be avoided.


