import sys
from pathlib import Path
import numpy as np, torch
C=Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0,str(C)); sys.path.insert(0,str(C.parent))
from floordiff.data import featurize, decode, load_validation_case
from floordiff import legalizer as lg
from floordiff.sample import load_checkpoint, rank_cost
diff=load_checkpoint(str(C/'floordiff/checkpoints/myrun/last.pt'), torch.device('cuda:0'))
tot={'rf':0,'af':0,'nok':0,'cases':0,'infeas':0}
for nb in [int(x) for x in sys.argv[1].split(',')]:
    case=load_validation_case(nb)
    n=len(case['area'])
    tensors,meta=featurize(case)
    b={k:v.unsqueeze(0).expand(4,*v.shape).contiguous().cuda() for k,v in tensors.items() if k not in('z0',)}
    with torch.no_grad():
        z=diff.sample(b['feat'],b['pair'],b['gfeat'],b['z_known'],b['freeze'],steps=50,seed=0)
    pred=decode(z[0].float().cpu(),meta).double()
    sol,info=lg.legalize_case(pred,case)
    g=info['graph']; tot['cases']+=1
    tot['rf']+=g['repair_failures']; tot['af']+=g['assign_failures']
    tot['nok']+=int(not g['final_assignment_ok']); tot['infeas']+=int(not info['hard']['feasible'])
    print(f"n={nb:3d} pre={int((case['cons'][:,1]!=0).sum()):2d} feas={info['hard']['feasible']} "
          f"repair_fail={g['repair_failures']} assign_fail={g['assign_failures']} final_ok={g['final_assignment_ok']}")
print(tot)
