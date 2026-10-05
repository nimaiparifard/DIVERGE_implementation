import os
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir, os.pardir))
print(repo_root)