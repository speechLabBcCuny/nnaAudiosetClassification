
python pipe1.py --input_files "/scratch/ebc327/nna/test/01 Itkillik/August 2016/shortfile.mp3" --abs_input_path "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --output_folder "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --segment_len "00:05:00" &> endlogs.txt &

python pipe2.py --input_files "/scratch/ebc327/nna/test/01 Itkillik/August 2016/shortfile.mp3" --abs_input_path "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --output_folder "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --segment_len "00:05:00"

python pipe3.py --input_files "/scratch/ebc327/nna/test/01 Itkillik/August 2016/shortfile.mp3" --abs_input_path "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --output_folder "/scratch/ebc327/nna/test/01 Itkillik/August 2016/" --segment_len "00:05:00"
# find /home/data/nna/stinchcomb/ -name "*.*3" -print0 | xargs -0 python end2end.py --input_files &> endlogs.txt &
srun -c32 find /scratch/ebc327/nna/NUI_DATA_copy -name "*.*3" | parallel -P 32 -N 1 python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '

find /scratch/ebc327/nna/NUI_DATA_copy -name "*.*3" | parallel -P 4 -N 20 --dryrun 'srun -N1 -c20 --exclusive echo {} | parallel -P 20  --dryrun -I  // python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" //'
seq 10 | parallel -P 4 -N 5 --dryrun 'srun -N1 -c20 --exclusive echo {} | parallel -P 20  --dryrun -I  // python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" //'

seq 10 | parallel -P 4 -N 5 'echo {} | parallel -P 2 -N 2 -I  // python pipe1.py --input_files // --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00"'
seq 10 | parallel -P 4 -N 5 'echo {} | parallel -N 5 -I  // --dryrun  echo {} //  '

seq 10 | parallel -P 4 -N 1 'seq {} | parallel -N 5 -I  //  echo {} - //  '

seq 10 | parallel -P 4 -N 1 'seq {} | parallel -N 5 -I  //  echo {} - //  '


find /scratch/ebc327/nna/NUI_DATA_copy -name "*.*3" | parallel -P5

find /scratch/ebc327/nna/NUI_DATA_copy -name "*.*3" -not -name "output*.*3" > mp3_files.txt
split -a 1 -d -l 50  mp3_files.txt segment
rm ./nna/segment*
mv segment* ./nna/
ls ./nna/segment* | wc -l

srun -N 1 -c20 --exclusive cat segment | parallel -P 20 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '
srun -N 1 -c20 --exclusive cat segment | parallel -P 20 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '
srun -N 1 -c20 --exclusive cat segment | parallel -P 20 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '
srun -N 1 -c20 --exclusive cat segment | parallel -P 20 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '
srun -N 1 -c20 --exclusive cat segmentae | parallel -P 20 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '

#############latest##################

#!/bin/bash
#SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=ebc327@nyu.edu
#SBATCH --mem=60GB
#SBATCH -t12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --array=0-9

module load parallel/20171022
module load cuda/10.1.105
module load anaconda3/5.3.0
. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh
conda activate HPCenv
conda deactivate
conda activate HPCenv
cd nna

srun cat segment${SLURM_ARRAY_TASK_ID} | parallel -P 10 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '

#############latest##################

&
srun -N 1 -n 10 -c1 --cpu_bind=cores cat segment${SLURM_ARRAY_TASK_ID} | parallel -P 10 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" ' &
srun -N 1 -n 10 -c1 --cpu_bind=cores cat segment${SLURM_ARRAY_TASK_ID} | parallel -P 10 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" ' &
srun -N 1 -n 10 -c1 --cpu_bind=cores cat segment${SLURM_ARRAY_TASK_ID} | parallel -P 10 -N 1 'python pipe1.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" ' &
wait
# https://www.hpc.kaust.edu.sa/tips/running-multiple-parallel-jobs-simultaneously

#####################

#############latestgpu##################

#!/bin/bash
#SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=ebc327@nyu.edu
#SBATCH --mem=30GB
#SBATCH -t1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATHC --gres=gpu:k80:1
#SBATCH --array=0-9

module load parallel/20171022
module load cuda/10.1.105
module load anaconda3/5.3.0
. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh
conda activate HPCenv
conda deactivate
conda activate HPCenv
cd nna

cat mp3_files.txt | shuf | parallel -P $SLURM_NTASKS 'srun -N1 -n1 --exclusive python pipe3.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '



cat mp3_files.txt | shuf | parallel -P 1 -N 10 'python pipe4.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '

srun cat mp3_files.txt | parallel -P 1 'python pipe3.py --input_files {} --abs_input_path "/scratch/ebc327/nna/NUI_DATA_copy/" --output_folder "/scratch/ebc327/nna/NUI_DATA_copy/" --segment_len "00:30:00" '

srun  -t04:00:00 --mem=30000  -c4 --gres=gpu:v100:1 --pty /bin/bash
srun  -t06:00:00 --mem=30000  -c4 --gres=gpu:v100:1 --pty /bin/bash
#############latestgpu##################


find /scratch/ebc327 -name "*_preprocessed" | wc -l
find /scratch/ebc327 -name "*rawembeddings.npy" | wc -l
find /scratch/ebc327 -name "*_preds.npy" | wc -l
