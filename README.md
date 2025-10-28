For empirical research (for myself only).

# Usage in run.ai clusters (A100) and B200.

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

## B200 nodes
```sh
kinit hnwong@UPENN.EDU
ssh hnwong@login.betty.parcc.upenn.edu
```
- home directory: `/vast/home/h/hnwong`
- find own jobs: `squeue -u $USER`
- run job: `srun --partition=dgx-b200   --pty --container-image=hnwong2025/base:latest   bash`
- more complicated one, need to find how to maps home addresses correctly:
```sh
srun --partition=dgx-b200 \
     --container-image=docker://hnwong2025/base:latest \
     --container-mounts=/vast/home/h/hnwong:/home/hnwong \
     --container-workdir=/home/hnwong \
     --container-env=HOME=/home/hnwong \
     --cpus-per-task=8 \
     --gpus=1 \
     --mem=128G \
     --pty \
     --time=01:00:00 \
     bash
```

## Docker commands:
```sh
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t hnwong2025/base:latest base
docker push hnwong2025/base:latest
```

# Git command:
Have the wrong commit (for example: adding large file) and want to modify it before pushing to remote:

- Keeps a copy of your current state in case anything goes wrong:
`git branch backup-before-rewrite`
- Prevents untracked local files from blocking rebase checkouts. `git stash -u -m "temp stash for rebase`
- Rebase to previous commit `git rebase -i HEAD~2` (# of steps depends on how many commits you have made)
- Change the line of the wrong commit to `edit`, and remove unnecessary commits afterwards
- At that commit, do things you want to, to fix the wrong commit. For example: for wrong large file adding `git rm --cached path/to/large_file`.
- Check this by `git rev-list --objects --all | grep 'path/to/large_file' || echo "✅ Large file removed from history`.
- Finally, `git push origin <branch>`.

For `git stash`

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

## Increase batch size

Two views, optimization and hardware

## Hyperparameter sweep experiments using wandb

## jax
TO-DO

# ssh stuff
