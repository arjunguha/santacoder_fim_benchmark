# SantaCoder Fill-in-the-Middle Benchmark

**This benchmark is now part of the [BigCode evaluation harness](https://github.com/bigcode-project/bigcode-evaluation-harness), which is the canonical way to run it.**

This is the fill-in-the-middle (FIM) benchmark that was used in the
SantaCoder and StarCoder papers. Those papers used the PSM format
exclusively. The code in this repository is further generalized to support
SPMv2 and variations used in StarCoder 2 ablations.

## Usage Example

The following commands will generate with the PSM and PSMv2 format 
and work on a 32GB GPU:

```
mkdir results
python3 generation.py --model-path $PATH_TO_STARCODERBASE_1B --output-dir results --batch-size 50 --mode PSM
python3 generation.py --model-path $PATH_TO_STARCODERBASE_1B  --output-dir results --batch-size 50 --mode SPMv2
```

This generates a CSV of results:

```
python3 evaluation.py results/*.jsonl > results.csv
```

This produces a handy plot:

```
python3 plot.py --input results.csv --output results.pdf
```
