# Infrastructure & Cluster Reference

Personal reference for compute cluster access, job submission, and DevOps workflows.

Large storage: `/shared_data0/hnwong` — check quota with `quota -vs`.

> DL experiment techniques, training templates, and JAX experiments are in
> [`../experiment_templates/`](../experiment_templates/).

---

## Compute Clusters

### run.ai — 4× A100

**Access:** `ssh hnwong@locust-login.seas.upenn.edu`, then `runai login` (username `hnwong@upenn.edu`).

**Submit an interactive job:**

```sh
runai submit honam \
   -i hnwong2025/base:latest \
   --attach --interactive --tty --stdin \
   -v /home/hnwong:/home/hnwong \
   -v /shared_data0:/shared_data0 \
   --cpu 8 -g 1 --large-shm --memory 128G \
   --working-dir /home/hnwong \
   -e HOME=/home/hnwong \
   --service-type=nodeport --port 30025:22 \
   -- /usr/sbin/sshd -D
```

Change port `30025` if running a second job.

**Common commands:**
- Delete job: `runai delete job honam`
- Port forward: `runai port-forward honam --port 30025:30025`
- Jupyter: create via run.ai UI
- TensorBoard: `runai port-forward honam --port 6006:6006`, then
  `ssh -L 6006:localhost:6006 hnwong@locust-login.seas.upenn.edu`

### Wharton Stats — 8× L40S

Uses the same locust login node. Launch via Slurm.

**Check resources:**

```sh
sinfo -o "%20N %20P %20G %20t"
squeue -o "%.10u %.18j %.8P %.6D %.10T %.12R %.20b"
```

**Batch job** (`sbatch whartonstats.sh`):

```sh
#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=moe

IMAGE_PATH="/shared_data0/hnwong/hnwong2025.sif"

CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
    --bind /shared_data0 \
    --bind /etc/pki \
    "$IMAGE_PATH" \
    uv run ... &
```

**Interactive job:**

```sh
srun --partition=whartonstat --gres=gpu:1 --mem=16GB --time=02:00:00 --pty bash
```

### B200 — PARCC (≈2.5× faster than A100)

**Access:**

```sh
kinit hnwong@UPENN.EDU
ssh hnwong@login.betty.parcc.upenn.edu
```

Home directory: `/vast/home/h/hnwong`

**Job commands:**
- Find own jobs: `squeue -u $USER`
- Cancel: `scancel <job_id>`
- Quick interactive: `srun --partition=dgx-b200 --pty --container-image=hnwong2025/base:latest bash`

**Full interactive job with home mount:**

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

---

## File Transfer

**run.ai ↔ B200:**

```sh
scp hnwong@locust-login.seas.upenn.edu:/shared_data0/hnwong/cache/file.npy \
    hnwong@login.betty.parcc.upenn.edu:~/.cache
```

---

## Docker

```sh
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t hnwong2025/base:latest base
docker push hnwong2025/base:latest
```

---

## Jupyter Notebook (inside server)

```sh
uv pip install ipykernel
uv run python -m ipykernel install --user --name=<project-name> --display-name="Python <project-name>"
```

---

## Hosted Platforms

[Modal](https://modal.com/) — serverless GPU compute.

```python
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .pip_install("torch")
    .add_local_python_source("models", "utils")
)

@app.function(image=training_image, gpu="A100-40GB", timeout=3600)
def get_results(params):
    return looper(params)

@app.local_entrypoint()
def main():
    for result in get_results.map(inputs):
        save_results(result)
```

---

## Torch Profiling

**Quick timing** (synchronize to avoid async CUDA misleading results):

```python
torch.cuda.synchronize(); t0 = time.time()
# ... operation ...
torch.cuda.synchronize(); elapsed = time.time() - t0
```

Ignore the first measurement (pre-loading overhead) and average over multiple runs.

**Full profiler with TensorBoard:**

```python
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

Install the viewer: `uv add torch_tb_profiler`, then launch:

```sh
uv run tensorboard --logdir=/shared_data0/hnwong/logs/profile --port=6006 --bind_all --load_fast=false
```

Port forwarding (same as TensorBoard above):
`runai port-forward honam --port 6006:6006`, then `ssh -L 6006:localhost:6006 hnwong@locust-login.seas.upenn.edu`.

---

## Git Recipes

**Select GPU:** `export CUDA_VISIBLE_DEVICES=1`

### Fix a wrong commit before pushing

```sh
git branch backup-before-rewrite              # safety copy
git stash -u -m "temp stash for rebase"        # stash untracked files
git rebase -i HEAD~2                           # adjust N to reach the bad commit
# Change the bad commit's line to "edit", remove unnecessary commits
# Fix the issue (e.g., git rm --cached path/to/large_file)
git rev-list --objects --all | grep 'path/to/large_file' || echo "✅ Removed"
git push origin <branch>
```

### Push to a private repository

```sh
git remote set-url origin https://Matheart:<token>@github.com/Matheart/<project>.git
```

### Sync a fork with upstream

```sh
git remote add upstream https://github.com/original/repo.git   # only once
git fetch upstream
git checkout main
git merge upstream/main   # or: git rebase upstream/main
git push origin main
```

### Useful commands

- `git stash` — temporarily save uncommitted changes
