import json
from pathlib import Path

path = Path('recordings/motions/put-away-1.json')
backup = path.with_suffix('.original.json')
data = json.loads(path.read_text())
samples = data['samples']
joints = [x for x in data.get('joints', []) if x != 'gripper']

if not backup.exists():
    backup.write_text(json.dumps(data, indent=2) + '\n')

start = 0
for i in range(1, len(samples)):
    delta = max(abs(samples[i][j] - samples[i - 1][j]) for j in joints)
    if delta > 0.015:
        start = max(0, i - 1)
        break
trimmed = [dict(x) for x in samples[start:]]

open_index = None
for i in range(1, len(trimmed)):
    if trimmed[i]['gripper'] - trimmed[i - 1]['gripper'] > 0.05:
        open_index = i
        break
if open_index is None:
    raise SystemExit('No gripper-opening transition found')
closed = -1.2
open_value = max(x['gripper'] for x in trimmed[open_index:])
for x in trimmed[:open_index]:
    x['gripper'] = closed
for x in trimmed[open_index:]:
    x['gripper'] = open_value

last_motion = open_index
for i in range(open_index + 1, len(trimmed)):
    delta = max(abs(trimmed[i][j] - trimmed[i - 1][j]) for j in joints)
    if delta > 0.015:
        last_motion = i
end = min(len(trimmed) - 1, last_motion + 5)
trimmed = trimmed[:end + 1]

t0 = trimmed[0]['t']
for x in trimmed:
    x['t'] = round(float(x['t'] - t0), 3)
data['samples'] = trimmed
data['duration'] = trimmed[-1]['t']
data['normalization'] = {'trimmed_lead_in': True, 'closed_until_drop': True,
                        'open_value': open_value, 'trimmed_tail': True}
path.write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps({'samples': len(trimmed), 'duration': data['duration'],
                  'open_index': open_index, 'closed': closed, 'open': open_value}))
