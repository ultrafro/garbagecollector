import asyncio
import json
import time
from autonomy.control_server import ControlHub, JOINTS
from autonomy.wrist_ik import WristIK


def test_capture_order_delete_persist_and_reject_stale(tmp_path):
    hub = ControlHub.__new__(ControlHub)
    hub.auto = hub.grab_active = hub.motion_recording = False
    hub.motion_task = None
    hub.placement_keypoints = []
    hub.keypoints_path = tmp_path / 'points.json'
    hub.wrist_ik = WristIK()
    hub.state = dict.fromkeys(JOINTS, 0.)
    hub.state['status'] = 'connected'
    hub.state_seen = time.monotonic()
    async def noop(*args, **kwargs): pass
    hub.send_autonomy_status = hub.grab_notice = noop
    async def run():
        for name in ['Lift', 'Drop']:
            await hub.browser_message({'type':'placement_capture','name':name})
        first, second = hub.placement_keypoints
        assert first['joints']['gripper'] == 0.
        hub.state['wrist_flex'] = .25
        await hub.browser_message({'type':'placement_update','id':first['id']})
        assert first['joints']['wrist_flex'] == .25
        assert first['name'] == 'Lift'
        assert json.loads(hub.keypoints_path.read_text())[0]['joints']['wrist_flex'] == .25
        await hub.browser_message({'type':'placement_move','id':second['id'],'direction':'up'})
        assert json.loads(hub.keypoints_path.read_text())[0]['name'] == 'Drop'
        await hub.browser_message({'type':'placement_delete','id':first['id']})
        hub.state_seen = 0
        await hub.browser_message({'type':'placement_capture','name':'Stale'})
        assert len(json.loads(hub.keypoints_path.read_text())) == 1
    asyncio.run(run())
