import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx

if __name__ == "__main__":
    src = "Datasets/Downloads"
    tar = "Datasets/train/DFWB"
    os.makedirs(tar, exist_ok=True)

    patch_size = 512
    overlap = 96
    p_max = 800

    def save_files(file_):
        path_contents = file_.split(os.sep)
        foldname = path_contents[-2]
        filename = os.path.splitext(path_contents[-1])[0]
        img = cv2.imread(file_)
        num_patch = 0
        w, h = img.shape[:2]
        if w > p_max and h > p_max:
            w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=np.int))
            h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=np.int))
            w1.append(w - patch_size)
            h1.append(h - patch_size)
            for i in w1:
                for j in h1:
                    num_patch += 1
                    patch = img[i : i + patch_size, j : j + patch_size, :]
                    savename = os.path.join(tar, foldname + "-" + filename + "-" + str(num_patch) + ".png")
                    cv2.imwrite(savename, patch)

        else:
            savename = os.path.join(tar, foldname + "-" + filename + ".png")
            cv2.imwrite(savename, img)

    files = []
    for dataset in ["DIV2K", "Flickr2K", "WaterlooED", "BSD400"]:
        df = natsorted(
            glob(os.path.join(src, dataset, "*.png"))
            + glob(os.path.join(src, dataset, "*.jpg"))
            + glob(os.path.join(src, dataset, "*.bmp"))
        )
        print(f"Found {len(df)} files in {dataset}")  # Debug print
        if len(df) > 0:
            print(f"First file found: {df[0]}")  # Debug print
        files.extend(df)

    print(f"Total files found across all datasets: {len(files)}")  # Debug print
    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = 1
    Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))
