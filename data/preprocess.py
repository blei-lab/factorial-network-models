import glob, pickle
import numpy as np
from scipy.io import loadmat

files = glob.glob("*.mat")
assert len(files) == 42, "Run download.sh to download the files first! " \
                         "There shouold be 42 .mat files in total."
As = []
for file in files:
    print("Loading ", file)
    data = loadmat(file)
    C = data["fibergraph"].toarray()

    # Convert to undirected binary adjacency matrix
    A = C > 0
    A = A | A.T
    np.fill_diagonal(A, False)
    As.append(A)
As = np.array(As)

# Save to pickle file
with open("kki-42-data.pkl", "wb") as f:
    pickle.dump(As, f)
