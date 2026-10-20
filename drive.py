"""Autonomous mode: the trained network steers, a simple controller holds speed.

Run AFTER train.py:
    python drive.py model.h5            # then open the simulator in Autonomous mode

The simulator sends a camera picture over a local connection (port 4567);
this script sends back (steering, throttle). The network only picks the
steering; throttle is deliberately kept dumb and constant.

IMPORTANT: images go through utils.preprocess() -- the exact same function
train.py uses, so the model always sees pictures in the format it learned.

WHY ENGINE.IO AND NOT SOCKET.IO: the simulator's built-in client speaks an
old dialect -- it opens the connection and starts streaming telemetry frames
without ever sending the socket.io "CONNECT" packet a modern socket.io
server waits for. A stock socket.io server therefore accepts the connection
and then silently drops every frame ("... is not connected to namespace").
Talking to the simulator at the engine.io layer (the transport underneath
socket.io) and handling the simple text packets by hand accepts both
dialects: clients that send CONNECT get an answer, clients that don't are
driven anyway. Verified against a simulated client in socktest/.
"""

import argparse
import base64
import io
import json

import engineio
import eventlet
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from utils import preprocess

eio = engineio.Server(async_mode="eventlet")
app = engineio.WSGIApp(eio, engineio_path="socket.io")

model = None
max_speed = 25.0
target_throttle = 0.3
steering_gain = 1.0
paved_gain = 1.3


def _road_saturation(img_rgb):
    """Mean saturation of the road patch ahead of the car.

    Grey asphalt (the paved lakeside section) has near-zero saturation;
    the dirt section is strongly orange. Measured on Armaan's recordings:
    sat < 0.10 is pavement, sat > 0.20 is dirt.
    """
    road = img_rgb[105:155, 100:220].astype(np.float32)
    r, g, b = road[..., 0].mean(), road[..., 1].mean(), road[..., 2].mean()
    mx, mn = max(r, g, b), min(r, g, b)
    return float((mx - mn) / mx) if mx else 0.0


def _send_packet(sid, packet):
    # engineio.send() wraps the payload in its own MESSAGE packet, so what we
    # pass here is the socket.io packet alone: "2[...]" for an event,
    # "0{...}" for the namespace-connect answer.
    eio.send(sid, packet)


def send_control(sid, steering_angle, throttle):
    payload = json.dumps(["steer", {"steering_angle": str(steering_angle),
                                     "throttle": str(throttle)}])
    _send_packet(sid, "2" + payload)


def handle_telemetry(sid, data):
    if data:
        img = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
        raw = np.asarray(img)
        x = preprocess(raw)
        pred = float(model.predict(x[None, ...], verbose=0)[0, 0])
        # v2 systematically understeers on the paved section (held-out
        # turning frames: predicted |s| 0.28 vs the human's 0.41; least
        # squares says x1.31) while its dirt steering is well scaled --
        # so boost ONLY on pavement, detected from the frame itself.
        gain = paved_gain if _road_saturation(raw) < 0.10 else steering_gain
        steering = float(np.clip(gain * pred, -1.0, 1.0))

        speed = float(data["speed"])
        throttle = target_throttle if speed < max_speed else 0.1

        print(f"steering: {steering:+.3f}   speed: {speed:.1f}   throttle: {throttle:.1f}", flush=True)
        send_control(sid, steering, throttle)
    else:
        _send_packet(sid, '2["manual",{}]')


@eio.on("connect")
def connect(sid, environ):
    print("simulator connected", flush=True)
    send_control(sid, 0.0, 0.0)


@eio.on("message")
def message(sid, data):
    if not isinstance(data, str):
        return
    if data.startswith("0"):
        # Client asked to open the socket.io namespace (socket.io packet
        # type 0): answer it, even though we happily drive clients that
        # never ask.
        _send_packet(sid, '0{"sid":"drive"}')
    elif data.startswith("2"):
        # socket.io EVENT packet: "2" followed by a JSON array.
        try:
            payload = json.loads(data[1:])
        except ValueError:
            return
        if isinstance(payload, list) and payload and payload[0] == "telemetry":
            handle_telemetry(sid, payload[1] if len(payload) > 1 else None)


@eio.on("disconnect")
def disconnect(sid):
    print("simulator disconnected", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="the .h5 file train.py saved")
    ap.add_argument("--speed", type=float, default=25.0, help="target speed")
    ap.add_argument("--throttle", type=float, default=0.3)
    ap.add_argument("--gain", type=float, default=1.0,
                    help="global steering multiplier")
    ap.add_argument("--paved-gain", type=float, default=1.3,
                    help="steering multiplier on the paved section "
                         "(v2 understeers there; measured optimum ~1.31)")
    args = ap.parse_args()
    steering_gain = args.gain
    paved_gain = args.paved_gain

    # compile=False: we only predict here, and it avoids a Keras 3 quirk where
    # loading a legacy .h5 tries to rebuild the training loss and fails.
    model = load_model(args.model, compile=False)
    max_speed = args.speed
    target_throttle = args.throttle

    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
