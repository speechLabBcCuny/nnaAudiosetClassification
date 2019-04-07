filename=/scratch/enis/data/nna/samples_wav/split_02:00/UMIAT_20130725_210243_2820m_00s__2880m_00s_14m_00s__16m_00s.wav
filename=/scratch/enis/data/nna/samples_wav/split_02:00/UMIAT_20130719_230628_2820m_00s__2880m_00s_50m_00s__52m_00s.wav
filename=/scratch/enis/data/nna/samples_wav/split_02:00/UMIAT_20130725_210243_1080m_00s__1140m_00s_54m_00s__56m_00s.wav rub
filename=/scratch/enis/data/nna/samples_wav/split_02:00/ROCKY_20160610_223054_2280m_00s__2340m_00s_08m_00s__10m_00s.wav music
filename=/scratch/enis/data/nna/samples_wav/split_02:00/UMIAT_20130725_210243_2820m_00s__2880m_00s_02m_00s__04m_00s.wav music
rsync -avz --progress Momentsnotice:$filename /Users/berk/Desktop/

path=/home/data/nna/stinchcomb/NUI_DATA/

find "$path" -type d -print0 | while read -d '' -r dir; do
    files=("$dir"/*)
    printf "%5d files in directory %s\n" "${#files[@]}" "$dir"
done
