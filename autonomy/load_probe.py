"""Guarded live wrist-ray extension with servo-load logging."""
import argparse, asyncio, csv, json, time
from pathlib import Path
import numpy as np
import websockets
from autonomy.wrist_ik import WristIK

PI='ws://192.168.0.119:8765'
JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']

async def run(max_mm=80., step_mm=2., hold=.25, load_delta=80., down_angle_deg=60.):
    if not 20 <= down_angle_deg <= 75: raise ValueError('down angle must be 20–75 degrees')
    ik=WristIK(); rows=[]; path=[]; latest_frame=None; folder=Path('screenshots');folder.mkdir(exist_ok=True)
    async with websockets.connect(PI) as ws:
        async def read_state():
            nonlocal latest_frame
            while True:
                raw=await asyncio.wait_for(ws.recv(),2)
                if isinstance(raw,bytes): latest_frame=raw; continue
                if isinstance(raw,str):
                    m=json.loads(raw)
                    if m.get('type')=='state': return m['data']
        hello=json.loads(await ws.recv())
        if not hello.get('state',{}).get('hardware'): raise RuntimeError('Pi did not report real hardware')
        state=hello['state']; initial={n:float(state[n]) for n in JOINTS}; baseline=np.array(state.get('servo_load',[0]*6),float)
        start=ik.fk(initial); original_ray=start[:3,2].copy(); horizontal=original_ray.copy();horizontal[2]=0
        horizontal/=np.linalg.norm(horizontal); angle=np.deg2rad(down_angle_deg); ray=horizontal*np.cos(angle)+np.array([0.,0.,-np.sin(angle)])
        current=initial; path.append(current)
        print(json.dumps({'start_xyz_m':start[:3,3].tolist(),'ray':ray.tolist(),'baseline_load':baseline.tolist(),'max_mm':max_mm,'load_delta':load_delta}),flush=True)
        try:
            await ws.send(json.dumps({'type':'stop'}))
            for mm in np.arange(step_mm,max_mm+step_mm/2,step_mm):
                target=start[:3,3]+ray*mm/1000
                try: command=ik.solve(current,start,0.,desired_position=target,position_tolerance=.006)
                except ValueError as exc: print('IK limit:',exc,flush=True);break
                command['gripper']=initial['gripper'];await ws.send(json.dumps({'type':'joints','enabled':True,'values':command}))
                deadline=time.monotonic()+hold; latest=None
                while time.monotonic()<deadline:
                    latest=await read_state(); await asyncio.sleep(.02)
                measured={n:float(latest.get(n,current[n])) for n in JOINTS};current=measured;path.append(current)
                measured_xyz=ik.fk(measured)[:3,3];loads=np.array(latest.get('servo_load',[0]*6),float);delta=loads-baseline
                frame_name='';
                if latest_frame:
                    frame_name=f'load-probe-frame-{len(rows):03d}.jpg';(folder/frame_name).write_bytes(latest_frame)
                row={'mm':float(mm),'xyz_error_mm':float(np.linalg.norm(measured_xyz-target)*1000),'max_load_delta':float(np.max(np.abs(delta))),'worst_load_joint':JOINTS[int(np.argmax(np.abs(delta)))],'frame':frame_name,**{f'load_{n}':float(loads[i]) for i,n in enumerate(JOINTS)},**{f'load_delta_{n}':float(delta[i]) for i,n in enumerate(JOINTS)}};rows.append(row)
                print(f"{mm:5.1f} mm  xyz error {row['xyz_error_mm']:5.1f} mm  loads {loads.astype(int).tolist()}  worst +{row['max_load_delta']:.0f} ({row['worst_load_joint']})",flush=True)
                if np.max(np.abs(delta))>=load_delta:
                    print('LOAD STOP: sustained collision candidate',flush=True);break
            print('Returning along measured path…',flush=True)
            for pose in reversed(path[:-1]):
                await ws.send(json.dumps({'type':'joints','enabled':True,'values':pose}));await asyncio.sleep(hold)
        finally:
            await ws.send(json.dumps({'type':'joints','enabled':True,'values':initial}));await asyncio.sleep(.4);await ws.send(json.dumps({'type':'stop'}))
            if rows:
                p=folder/f'load-probe-{time.time_ns()}.csv'
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                (folder/'load-probe-latest.json').write_text(json.dumps({'rows':rows,'joints':JOINTS,'baseline_load':baseline.tolist(),'created':time.time()}))
                print('LOG',p,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--max-mm',type=float,default=80);ap.add_argument('--load-delta',type=float,default=80);ap.add_argument('--down-angle-deg',type=float,default=60);args=ap.parse_args();asyncio.run(run(args.max_mm,load_delta=args.load_delta,down_angle_deg=args.down_angle_deg))
