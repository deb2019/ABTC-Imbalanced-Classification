import glob, nbformat
try:
    from nbformat.validator import normalize  # nbformat >= 5.1.4
except ImportError:
    normalize = None

paths = glob.glob(r'notebooks\**\*.ipynb', recursive=True)
for p in paths:
    nb = nbformat.read(p, as_version=nbformat.NO_CONVERT)
    if normalize:
        normalize(nb)  # add missing ids & normalize metadata
    nbformat.write(nb, p)
    print("normalized:", p)
print("Done.")
