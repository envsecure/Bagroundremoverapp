import kagglehub
import shutil


path = kagglehub.dataset_download("nikhilroxtomar/person-segmentation")


dest = '/teamspace/studios/this_studio/bg_remover/data/raw'
shutil.copytree(path, dest, dirs_exist_ok=True)
print("Data moved to:", dest)