import os
import shutil
import uuid
import pathlib
from git import Repo
import urllib.request
from pathlib import Path
import zipfile

DATASETs = ['assist2009',
             'assist2017',
             'AKT_assist2017']

DATASET_URLs = {
    # Add more datasets as needed
}

GIT_REPO = 'https://github.com/badranx/KTdata'

dataset2gitbranch = {
    'AKT_assist2017': 'data/AKT_assist2017',
    'assist2009': 'data/assist2009',
    'corr_assist2009': 'data/corr_assist2009',
    'corr2_assist2009': 'data/corr2_assist2009',
    'duolingo2018_es_en': 'data/duolingo2018_es_en',
    'riiid2020': 'data/riiid2020',
    'algebra2005': 'data/algebra2005'
}

dataset2stdkcs = {
    'duolingo2018_es_en': 1.0
}

def gitdownload(dataset_name, download_path):
    try: 
        branch = dataset2gitbranch[dataset_name]
    except KeyError as e:
        raise Exception('{} dataset is not available for download!'.format(dataset_name)) from e

    tmpdir = './ktbench_' + str(uuid.uuid4())
    Repo.clone_from(GIT_REPO, tmpdir, b=branch, depth=1, single_branch=True)
    #zip_file_path =  tmpdir/ f"{dataset_name}.zip"
    flist = [p for p in pathlib.Path(tmpdir).iterdir() if p.is_file()]
    zip_file_path = flist[0]

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        extracted = zip_ref.namelist()
        zip_ref.extractall(download_path)
    print(f"{dataset_name} dataset downloaded successfully.")

    shutil.rmtree(tmpdir)
    extracted_file = os.path.join(download_path, extracted[0])
    return extracted_file

def download_dataset(dataset_name, download_path):

    if dataset_name not in DATASET_URLs:
        raise Exception("{dataset_name} does not have a direct download link")

    dataset_url = DATASET_URLs[dataset_name]
    download_path = Path(download_path)

    zip_file_path = download_path / f"{dataset_name}.zip"

    download_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_name} dataset...")
    urllib.request.urlretrieve(dataset_url, zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        extracted = zip_ref.namelist()
        zip_ref.extractall(download_path)
    print(f"{dataset_name} dataset downloaded successfully.")

    os.remove(zip_file_path)
    extracted_file = os.path.join(download_path, extracted[0])
    return extracted_file

if __name__ == "__main__":
    #dspath = download_dataset('AKT_assist2017', '.')
    dspath = gitdownload('AKT_assist2017', '.')
    print(dspath)
