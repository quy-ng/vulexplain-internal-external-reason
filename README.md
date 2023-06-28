# vulexplain-internal-external-reason
## Up and Run
```console
conda env create -f binder/environment.yml
conda activate vul-intext-reason
```
## Update Environment after adding
```console
conda env update --file binder/environment.yml --prune
```
## Export Environment
```console
conda env export --from-history -f binder/environment.yml
```
