import asyncio
import cv2
import numpy as np
from autonomy.control_server import ControlHub


def test_exposure_adapts_both_directions_with_bounds():
    hub = ControlHub.__new__(ControlHub)
    hub.camera_exposure, hub.camera_gain, hub.camera_adjusted = 30, 0, 0.
    sent = []
    async def send(message):
        sent.append(message)
    hub.pi_send = send
    async def frame(brightness):
        hub.camera_adjusted = 0.
        _, jpeg = cv2.imencode('.jpg', np.full((100, 100), brightness, np.uint8))
        await hub.adjust_camera(jpeg.tobytes())
    async def run():
        await frame(255)
        assert hub.camera_exposure < 30
        await frame(0)
        assert hub.camera_exposure > 15
        for _ in range(20):
            await frame(0)
        assert (hub.camera_exposure, hub.camera_gain) == (300, 60)
        for _ in range(30):
            await frame(255)
        assert (hub.camera_exposure, hub.camera_gain) == (1, 0)
        await frame(115)
        assert (hub.camera_exposure, hub.camera_gain) == (1, 0)
    asyncio.run(run())
    assert all(m['type'] == 'camera_settings' for m in sent)


def test_settings_are_only_sent_when_they_change():
    """Re-sending identical settings made the Pi re-run v4l2-ctl every second
    against the device ffmpeg was streaming from, stalling the video."""
    import asyncio
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from autonomy.control_server import ControlHub

    hub = ControlHub.__new__(ControlHub)
    hub.camera_exposure, hub.camera_gain, hub.camera_adjusted = 30, 0, 0.
    sent = []

    async def pi_send(message):
        sent.append(message)
    hub.pi_send = pi_send

    # A well-exposed frame: nothing to correct, so nothing to send.
    frame = np.full((360, 640), 115, np.uint8)
    jpeg = cv2.imencode('.jpg', frame)[1].tobytes()

    for _ in range(5):
        hub.camera_adjusted = 0.        # defeat the 1 s rate limit
        asyncio.run(hub.adjust_camera(jpeg))
    assert sent == [], f'unchanged settings must not be re-sent: {sent}'

    # A dark frame does need a change, and that must still go out.
    dark = cv2.imencode('.jpg', np.full((360, 640), 20, np.uint8))[1].tobytes()
    hub.camera_adjusted = 0.
    asyncio.run(hub.adjust_camera(dark))
    assert len(sent) == 1 and sent[0]['type'] == 'camera_settings'


def test_unchanged_settings_are_resent_every_10_s_in_case_the_pi_reset():
    hub = ControlHub.__new__(ControlHub)
    hub.camera_exposure, hub.camera_gain, hub.camera_adjusted = 300, 60, 0.
    sent = []
    async def send(message):
        sent.append(message)
    hub.pi_send = send
    _, jpeg = cv2.imencode('.jpg', np.full((100, 100), 20, np.uint8))    # dark, but already at the limit
    async def run():
        await hub.adjust_camera(jpeg.tobytes())
        assert sent == []                                              # nothing new to say yet
        hub.camera_sent -= 11.
        hub.camera_adjusted = 0.
        await hub.adjust_camera(jpeg.tobytes())
    asyncio.run(run())
    assert sent == [{'type': 'camera_settings', 'exposure': 300, 'gain': 60}]
