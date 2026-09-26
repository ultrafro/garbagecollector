"""Offline GPU benchmark; contains no robot connection or motion code."""
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

INDICES = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 126]
LABELS = ['plastic wrapper', 'food packaging', 'plastic bottle', 'can', 'rug']


def run(args):
    assert torch.cuda.is_available(), 'GPU environment required'
    output = args.folder/'strong-models'
    output.mkdir(exist_ok=True)
    samples = [args.folder/f'raw-{i:05d}.jpg' for i in INDICES]
    name = args.model
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    if name.startswith('yoloe'):
        from ultralytics import YOLOE
        model = YOLOE(f'{name}.pt')
        model.set_classes(LABELS)

        def predict(path):
            result = model.predict(str(path), device=0, imgsz=640, conf=.05, verbose=False)[0]
            return [{'label': result.names[int(b.cls.item())], 'score':float(b.conf.item()), 'box':b.xyxy[0].tolist()} for b in result.boxes], None

    elif name == 'dino':
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = 'IDEA-Research/grounding-dino-base'
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to('cuda').eval()
        text = 'a plastic wrapper. food packaging. a plastic bottle. a can. a rug.'

        def predict(path):
            image = Image.open(path).convert('RGB')
            inputs = processor(images=image, text=text, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                prediction = model(**inputs)
            result = processor.post_process_grounded_object_detection(prediction, inputs.input_ids, threshold=.20, text_threshold=.20, target_sizes=[image.size[::-1]])[0]
            labels = result.get('text_labels', result.get('labels'))
            return [{'label':str(label), 'score':float(score), 'box':box.tolist()} for box, score, label in zip(result['boxes'],result['scores'],labels)], None

    else:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        model_id = 'Qwen/Qwen3-VL-4B-Instruct'
        processor = AutoProcessor.from_pretrained(model_id)
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, quantization_config=config, device_map={'':'cuda:0'}, torch_dtype=torch.float16, attn_implementation='sdpa').eval()
        prompt = ('Locate visible discarded packaging or litter that a robot could pick up. Include the WHOLE object, not its printed logo. '
                  'Do not label the rug, floor, shadows, or patterns as trash. If only rug is visible, return an empty list. '
                  'Return only minified single-line JSON with no spaces or code fences: {"objects":[{"label":"plastic wrapper","box":[x1,y1,x2,y2]}]}. '
                  'Use integer bounding-box coordinates normalized to 0-1000. Include partially visible trash if present.')

        def predict(path):
            messages = [{'role':'user','content':[{'type':'image','image':str(path.resolve())},{'type':'text','text':prompt}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=180, do_sample=False)
            raw = processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            try:
                parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
                objects = parsed['objects']
                width, height = Image.open(path).size
                boxes = []
                for item in objects:
                    box = item['box']
                    if len(box) != 4 or not all(isinstance(v,(int,float)) and 0 <= v <= 1000 for v in box):
                        raise ValueError('Invalid coordinates')
                    boxes.append({'label':item['label'],'score':None,'box':[box[0]*width/1000,box[1]*height/1000,box[2]*width/1000,box[3]*height/1000]})
                return boxes, raw
            except (ValueError, KeyError, TypeError):
                return [], {'parse_error':True, 'raw':raw}

    load_seconds = time.perf_counter()-load_start
    print(json.dumps({'loaded':name,'seconds':load_seconds}),flush=True)
    # Warmup excluded from measured latency.
    predict(samples[0])
    rows = []
    thumbs = []
    for path in samples:
        torch.cuda.synchronize()
        start = time.perf_counter()
        boxes, raw = predict(path)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        image = cv2.imread(str(path))
        for item in boxes:
            x1,y1,x2,y2 = [int(x) for x in item['box']]
            color = (180,180,180) if 'rug' in item['label'] else (0,255,255)
            cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            cv2.putText(image, item['label'][:34]+('' if item['score'] is None else f" {item['score']:.2f}"), (max(0,x1),max(18,y1)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.putText(image,f'{name}: {path.name} {elapsed:.0f}ms',(8,345),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,0),1)
        cv2.imwrite(str(output/f'{name}-{path.stem}.jpg'),image)
        thumbs.append(image)
        rows.append({'file':path.name,'latency_ms':elapsed,'boxes':boxes,'raw_response':raw})
        report={'model':name,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'load_seconds':load_seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'mean_ms':float(np.mean([r['latency_ms'] for r in rows])),'rows':rows}
        (output/f'{name}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(rows[-1]),flush=True)
    sheet = np.concatenate([np.concatenate(thumbs[i:i+3],axis=1) for i in range(0,len(thumbs),3)],axis=0)
    cv2.imwrite(str(output/f'{name}-sheet.jpg'),sheet)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path('recordings/targeting-20260926-104256'))
    parser.add_argument('--model',choices=['yoloe-26m-seg','yoloe-26l-seg','dino','qwen'],required=True)
    run(parser.parse_args())
