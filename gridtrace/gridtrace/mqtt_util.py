"""paho-mqtt version shim, so the same code works on paho 1.x and 2.x."""
from __future__ import annotations

import paho.mqtt.client as mqtt


def make_client(client_id: str):
    try:                                   # paho 2.x requires an explicit callback API version
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:                 # paho 1.x
        return mqtt.Client(client_id=client_id)
