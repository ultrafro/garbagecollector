"""Score the small hand-annotated approach clip, with explicit caveats."""
import json
from pathlib import Path


def iou(a, b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union else 0.


def main():
    folder = Path('recordings/targeting-20260926-104256')
    labels = json.loads((folder/'evaluation-labels.json').read_text())
    result = {'caveat':labels['note'], 'criterion':'Best non-background box IoU >= 0.5 against approximate whole-wrapper annotation. Different model confidence scales are not comparable.', 'models':{}}
    for path in (folder/'strong-models').glob('*.json'):
        report = json.loads(path.read_text())
        if 'rows' not in report:
            continue
        scored=[]
        for row in report['rows']:
            truth=labels['frames'][row['file']]
            boxes=[b for b in row['boxes'] if not any(word in b['label'].lower() for word in ['rug','carpet','floor'])]
            best=max((iou(b['box'],truth['box']) for b in boxes),default=0.) if truth['box'] else None
            scored.append({'file':row['file'],'status':truth['status'],'best_iou':best,'non_background_boxes':len(boxes),'parse_error':isinstance(row.get('raw_response'),dict) and row['raw_response'].get('parse_error',False)})
        result['models'][report['model']]={
            'visible_hits':sum(r['status']=='visible' and r['best_iou']>=.5 for r in scored),
            'visible_total':sum(r['status']=='visible' for r in scored),
            'partial_hits':sum(r['status']=='partial' and r['best_iou']>=.5 for r in scored),
            'partial_total':sum(r['status']=='partial' for r in scored),
            'empty_false_positive_frames':sum(r['status']=='absent' and r['non_background_boxes']>0 for r in scored),
            'empty_total':sum(r['status']=='absent' for r in scored),
            'parse_errors':sum(r['parse_error'] for r in scored),
            'mean_ms':report['mean_ms'],'peak_allocated_GiB':report['peak_allocated_GiB'],'frames':scored}
    (folder/'strong-models'/'scores.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
