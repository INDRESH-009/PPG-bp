from tqdm import tqdm

def tbar(iterable, desc):
    return tqdm(iterable, desc=desc, ncols=100, leave=False)
