from pathlib import Path
import shutil
import os
import os.path


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")
    if "paddlenlp" in path:
        return
    for item in os.listdir(path):
        if '.git' not in item:
            print("|      " * depth + "|--" + item)

            newitem = path + '/' + item
            if os.path.isdir(newitem):
                dfs_showdir(newitem, depth + 1)


# for p in Path(".").glob("**/__pycache__"):
#     shutil.rmtree(p)
if __name__ == '__main__':
    dfs_showdir('./', 0)
