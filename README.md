# vulexplain-internal-external-reason
## With Docker
```console
./run_docker.sh
```
## Without Docker
```console
conda env create -f binder/environment.yml
conda activate vul-intext-reason
```

Install other dependencies in OS
+ Ubuntu clang-format version 14.0.0-1ubuntu1
+ Graphviz
```console
sudo apt install clang-format
sudo apt install graphviz
```

## Update Environment after adding
```console
conda env update --file binder/environment.yml --prune
```
## Export Environment
```console
conda env export --from-history -f binder/environment.yml
```


## Encounter issue with GLIBCXX not found
Firstly, `find / -name "libstdc++.so*"` and create a symbolic link properly

# Run CodeT5+ k=10, k=5
```console
./run_t5p_new.sh
```
# Run CodeBert-based
```console
./run_bert_seq2seq_new.sh
```
# Run CodeT5+ with k=(5 10 15 20 25 30 40 50 60 70 80 90)
```console
./run_t5p_percentage.sh
```
