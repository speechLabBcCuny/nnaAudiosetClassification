#!/bin/bash

# Copy wav files in a directory tree to another directory tree of flac
# files. If third and fourth arguments (K and N) are provided, then
# take every Nth filename starting at position K (for parallelization
# across several simultaneous runs).

if [ $# -lt 3 -o $# -gt 5 ]; then
    echo "Usage: `basename $0` src_dir dst_dir [k n]"
    exit 1
fi

SRC=$1
DST=$2
K=${3:-1}
N=${4:-1}


find "$SRC" -name '*.wav' \
    | sort \
    | sed -n "${K}~${N}p" \
    | while read fullwav ; do 
    wav="${fullwav#${SRC}}"
    outfile="${DST}/${wav%.wav}.flac"
    if [ -f "$outfile" ] ; then
        echo "Skipping $outfile"
    else
        subdir=`dirname "$wav"`
        mkdir -p "${DST}/${subdir}"
        ffmpeg -n -nostdin -i "$fullwav" "$outfile"
    fi
done
