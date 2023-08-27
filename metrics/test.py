import os
import pickle
import hashlib
import glob
from typing import Any




def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache_dir is not None:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            return open(cache_files[0], "rb")

def open_pickle(pickle_path):
    with open_url(pickle_path) as f:
        data = pickle.load(f, encoding='latin1')
    return data


inception = open_url('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn') # inception_v3_features.pkl
print(inception)