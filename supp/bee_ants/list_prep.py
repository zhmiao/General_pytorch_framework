# %%
import os
from glob import glob

# %%
data_root = './hymenoptera_data'
tr = glob(os.path.join(data_root, 'train', '**/*.jpg'), recursive=True)
val = glob(os.path.join(data_root, 'val', '**/*.jpg'), recursive=True)

# %%
tr_list = open('./lists/train.txt', 'w')
for i in tr:
    label = 0 if 'ants' in i else 1
    tr_list.write('{} {}\n'.format(i.split('/', 2)[-1], label))
tr_list.close()

val_list = open('./lists/val.txt', 'w')
for i in val:
    label = 0 if 'ants' in i else 1
    val_list.write('{} {}\n'.format(i.split('/', 2)[-1], label))
val_list.close()
# %%

