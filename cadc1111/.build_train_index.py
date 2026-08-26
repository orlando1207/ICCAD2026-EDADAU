"""One-off: index floorset_lite training files by block count -> .train_index.json"""
import glob, json, os, sys, torch

import stress                                   # for the data root
FILES = sorted(glob.glob(os.path.join(stress.DATA, 'floorset_lite',
                                      'worker_*', 'layouts_*.th')))
idx = {}
for k, p in enumerate(FILES):
    try:
        d = torch.load(p, weights_only=False)
        area = d[0][0][:, 0]
        n = int((area != -1).sum())
        # confirm every layout in the file has the same n
        same = all(int((d[0][i][:, 0] != -1).sum()) == n for i in (1, 55, 111))
    except Exception as e:
        print('skip', p, e); continue
    idx.setdefault(str(n), []).append([p, 112 if same else 1])
    if k % 500 == 0:
        print(k, len(FILES), flush=True)
json.dump(idx, open(stress.TRAIN_INDEX, 'w'))
print('n values:', sorted(int(x) for x in idx), 'files', sum(len(v) for v in idx.values()))
