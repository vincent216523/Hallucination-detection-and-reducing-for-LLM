#intearctive job
srun --partition normal --account=XXX --nodes=1 --gpus-per-node=1  --time 0-00:30:00 --pty bash

# batch job
sbatch msbd5002/job0-500_data.sh --partition normal --account=XXX --username csauac
sbatch msbd5002/job500-1500.sh --partition normal --account=XXX --username csauac
sbatch msbd5002/job1500-2500.sh --partition normal --account=XXX --username csauac


# check queue
squeue -u csauac

#  cancel job
scancel <job_id>

# view .out
cat job-<job_id>.out

# watch .out
tail -f job-<job_id>.out
