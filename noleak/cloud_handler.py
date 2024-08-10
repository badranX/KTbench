from git import Repo
import json
from pathlib import Path
import uuid


import os
import shutil

def copy_directory(source_dir, destination_dir):
    try:
        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir, ignore_errors=True)

        shutil.copytree(source_dir, destination_dir)
        shutil.rmtree(destination_dir/'.git', ignore_errors=True)
    except Exception as e:
        print(f"Error: {e}")

class  GitStore:
    def __init__(self, branch_name):
        self.branch_name = branch_name
        self.is_ready = False
        self.tmpdir = Path('./ktbench_' + str(uuid.uuid4()))
        repofile = Path.cwd() / 'gitinfo.json'
        if repofile.is_file():
            with open(repofile, 'r') as f:
                self.gitinfo = json.load(f)
                self.remote_url = self.gitinfo['remote_url']
                self.local_dir = self.gitinfo['local_dir']
                self.local_dir = Path(self.local_dir)

            #Repo.clone_from(self.remote_url, self.tmpdir, b='main')#, depth=1, single_branch=True)
            self.repopath = Path(self.local_dir/self.tmpdir)
            self.repo = Repo.init(self.repopath)

            # Add a remote named 'origin' with the provided URL
            self.origin = self.repo.create_remote('origin', self.remote_url)

            self.is_ready = True


    def git_store_dir(self, dirpath):

        if not self.is_ready:
            return None
     # Initialize a new Git repository in the provided directory path
        dirpath = Path(dirpath)
        copy_directory(dirpath, self.repopath/dirpath.name)

        # Add all files to the staging area
        #repo.index.add(".")
        self.repo.index.add(dirpath.name)

        # Commit the changes
        self.repo.index.commit("add {}".format(dirpath.name))


        # Push the content forcefully to the remote repository
        #self.origin.push('master', force=True, u=True)
        self.origin.push(refspec='HEAD:' + self.branch_name, force=True)


if __name__ == "__main__":
    s = GitStore('del/test')
    s.git_store_dir('./del')
    s.git_store_dir('./del')
