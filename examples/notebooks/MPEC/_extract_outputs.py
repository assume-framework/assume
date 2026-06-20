import json
with open('12_eval_02a_elasticDemand_EE_Ramp.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    src = ''.join(cell.get('source',[]))
    if 'Run MPEC + UC for all non-renewable' in src:
        for o in cell.get('outputs',[]):
            if o.get('text'):
                print(''.join(o['text']))
        break
