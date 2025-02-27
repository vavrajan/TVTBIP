import os

data = 'hein-daily'

### First set up directories on the cluster:
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, 'data')
slurm_dir = os.path.join(project_dir, 'slurm', data)
out_dir = os.path.join(project_dir, 'out', data)
err_dir = os.path.join(project_dir, 'err', data)
python_dir = os.path.join(project_dir, 'code', 'tbip')

if not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)


# For now just use the environment for testing.
# partition = 'gpu-test'
partition = 'gpu'

### A dictionary of scenarios to be explored
# Default values correspond with the classical TBIP model (with topic-specific locations) with gamma and CAVI updates.
# List only the FLAGS that you want to be changed.

# First scenario
scenarios = {}
for s in range(97, 115):
    if s == 97:
        pip = 'PF'
    else:
        pip = 'previous'
    scenarios[str(s)] = {"data": data+"-"+str(s),
                         "previous_data": data+"-"+str(s-1),
                         "epsilon": 1e-08,
                         "learning_rate": 0.01,
                         "pre_initialize_parameters": pip,
                         "max_steps": 1000,   # 300000,
                         "num_topics": 25,
                         "batch_size": 512,
                         }

### Creating slurm files and one file to trigger all the jobs that continue from previous session to the next (98-114)
with open(os.path.join(slurm_dir, 'run_all_97_114.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    for name in scenarios:
        flags = ''
        for key in scenarios[name]:
            flags = flags+'  --'+key+'='+str(scenarios[name][key])
        with open(os.path.join(slurm_dir, name+'.slurm'), 'w') as file:
            file.write('#!/bin/bash\n')
            file.write('#SBATCH --job-name=' + data + ' # short name for your job\n')
            file.write('#SBATCH --partition='+partition+'\n')

            # Other potential computational settings.
            # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
            # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
            # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
            # file.write('#SBATCH --mem=1G # total memory per node\n')
            file.write('#SBATCH --mem-per-cpu=356000 # in megabytes, default is 4GB per task\n')
            # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
            # file.write('#SBATCH --mail-type=ALL\n')

            file.write('#SBATCH -o '+os.path.join(out_dir, '%x_%j_%N.out')+'      # save stdout to file. '
                                             'The filename is defined through filename pattern\n')
            file.write('#SBATCH -e '+os.path.join(err_dir, '%x_%j_%N.err')+'      # save stderr to file. '
                                             'The filename is defined through filename pattern\n')
            file.write('#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)\n')
            file.write('\n')
            file.write('. /opt/apps/2023-04-11_lmod.bash\n')
            file.write('ml purge\n')
            file.write('ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn\n')
            file.write('\n')
            file.write('cd /home/jvavra/TVTBIP/\n')
            # file.write('conda activate tf_TBIP\n')
            file.write('conda activate tf_1_15\n')
            file.write('\n')
            file.write('python '+os.path.join(python_dir, 'tbip_different_init.py')+flags+'\n')
        # Add a line for running the batch script to the overall slurm job.
        all_file.write('sbatch --dependency=singleton '+os.path.join(slurm_dir, name+'.slurm'))
        all_file.write('\n')
