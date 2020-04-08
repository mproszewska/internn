"""
Common functions for handling models.
"""
import os
import sys
import tarfile
import zipfile

from pathlib import Path
from urllib.request import urlretrieve

def download(url, dest_dir):
    """
    Download and extract the data if it doesn't already exist.

    PARAMETERS
    ----------
    url : str
        URL for to download tar-file.
    dest_dir : str, optional
        Destination folder for the downloaded file.
    """
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_dir, filename)

    if not os.path.exists(file_path):

        def download_progress_hook(count, blockSize, totalSize):
            percent = float(count * blockSize / totalSize)
            sys.stdout.write("\rDownload progress: {0:.0%}".format(percent))
            sys.stdout.flush()

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        new_file_path, _ = urlretrieve(
            url=url, filename=file_path, reporthook=download_progress_hook
        )

        print("\nDownload finished. Extracting files.")

        if new_file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(dest_dir)
        elif new_file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(dest_dir)
        else:
            print("Cannot extract. Unsupported extension of downloaded file.")
            return

        print("Done.")