import json
with open('/scratch/anavekar/KernelGYM/results/drkernel14b_new_pipeline_raw.json') as f:
    d = json.load(f)
for p in d['results'][:3]:
    print('===', p['problem_id'], '===')
    for t in p['turn_history']:
        print(f'  Turn {t["turn"]}: compiled={t["compiled"]}, correct={t["correctness"]}')
        if t.get('error_message'):
            print(f'    ERROR: {t["error_message"]}')
