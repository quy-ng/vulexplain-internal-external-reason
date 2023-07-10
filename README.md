# vulexplain-internal-external-reason
## Up and Run
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
