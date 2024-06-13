#!/bin/bash
set -euo pipefail

# run matplotlib once to generate the font cache
python -c "import matplotlib as mpl; mpl.use('Agg'); import pylab as plt; fig, ax = plt.subplots(); fig.savefig('test.png')"
test -e test.png && rm test.png
# icse package needs it
python -c "import nltk; nltk.download('punkt')"