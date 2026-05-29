# PAIL: Positional Algorithm and Index for Lightweight filtering

An efficient algorithm for the computation of kNN queries on sets.

## Compilation

### Run using Ansible

This section shows how PAIL can be installed using the provided Ansible scripts.
The following assumptions are made on the machine:
* Up-to-date version of Debian (version 12 or higher, Debian 13 is confirmed to work)
* Root permissions
* The following standard dependencies are installed (using `apt install`):
    * git
    * ansible-core

1. Clone the repository
```
git clone https://github.com/DatabaseGroup/setknn.git setknn-src
cd setknn-src
```

2. Run the `knn` Ansible script/role on the local machine
```
ansible-playbook -i "localhost," -c local --extra-vars "home_dir=${HOME}" ansible/knn.yml --skip-tags datasets,experiments
```

3. Executables are copied to `${HOME}/knn/bin`
```
cd ${HOME}/bin
```

## Usage

PAIL offers a command-line interface to configure data input, executed algorithm, algorithm parameters, sample size, etc.

```
USAGE:
  -f [ --input-file ] arg               Input file
  -k [ --top-k ] arg                    result size
  -a [ --algorithm ] arg (=slim)        One of the following: baseline, 
                                        baselinepp, topkbaseline, slim, full, 
                                        transformation, partition, palloc, les3,
                                        puffinn
  --les3-groups arg                     Path to a precomputed LES3 group file
  --puffinn-memory-bytes arg            PUFFINN memory budget in bytes
  --puffinn-recall arg                  PUFFINN recall target
  -c [ --concurrency ] arg (=16)        Level of concurrency/number of threads.
                                        -1 for all cores
  -s [ --sample-size ] arg (=10000)     Sample size for sampled join
  -t [ --timeout ] arg (=0)             Query timeout in seconds. 0 disables it
  -r [ --seed ] arg (=1111638594)       Seed for prng
  -m [ --mode ] arg (=sample)           Sample or join
  -x [ --suffix ] arg (=0)              suffix depth for filtering
  -v [ --vector ] arg (=0)              Activate filtering using vectors
  -l [ --label ] arg                    Label for the runs (printed in json)
  -g [ --length-grouping ] arg (=identity)
                                        Select length grouping (identity, 
                                        wsqrt, usqrt, aio)
  -d [ --deletion ]                     Enable deletion signatures for 
                                        partition
```

### Description of Arguments

- `-a`: Algorithm to be executed.
  - `slim` activates eager horizontal scanning.
  - `full` does not use eager horizontal scanning.
  - `les3` runs LES3 kNN search using a precomputed group file supplied with `--les3-groups`.
  - `puffinn` runs approximate Jaccard kNN search using PUFFINN. It requires `--puffinn-memory-bytes`,
    `--puffinn-recall`, and `-c 1`.
- `-l`: Length grouping. Only has an effect for `Full` and `Slim`.
    - `identity` disables grouping, one group is created for each size
    - `aio` puts all sets into a single group as in `topkbaseline`
    - `wsqrt` performs weighted grouping based on the token frequency, this is the self-join scenario described in the paper
    - `usqrt` assumes a uniform distribution of query tokens, this is the R-S-join scenario in the paper

## Dataset Format

A line corresponds to a set. Each set consists of numerical tokens separated by spaces.
Sets are assumed to be already preprocessed according to the following properties:

- Sets are ordered in increasing size
- Each token appears inside a set at most once, i.e., we do not have multisets
- The tokens are in increasing value
- The token values are assigned in inverse global frequency, i.e., 1 (or 0) is the least frequent token and so on.

An implementation of such preprocessing is available [here](https://frosch.cosy.sbg.ac.at/datasets/sets/tools).

**Example** (3 sets):
```
1 2 4
4 5 42 105
5 6 38 42 105
```
