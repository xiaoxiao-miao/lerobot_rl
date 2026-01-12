#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
import time
from typing import Any

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.xr.config_xr import XRTeleopConfig
from lerobot.utils.rotation import Rotation


class XRTeleop(Teleoperator):
    """Teleoperator that consumes XR pose streams over WebSocket."""

    config_class = XRTeleopConfig
    name = "xr"

    def __init__(self, config: XRTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_payload: dict[str, Any] | None = None
        self._payload_lock = threading.Lock()
        self._websocket = None
        self._connected = False

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for part in self.config.pose_paths:
            features[f"{self.config.output_prefix}.{part}.pos"] = np.ndarray
            features[f"{self.config.output_prefix}.{part}.rot"] = Rotation
        if self.config.include_raw:
            features[f"{self.config.output_prefix}.raw"] = dict
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        if self._listener_thread and self._listener_thread.is_alive():
            return
        self._stop_event.clear()
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()

    def _listen_loop(self) -> None:
        url = self._build_url()
        while not self._stop_event.is_set():
            try:
                with connect(url) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    for message in websocket:
                        if self._stop_event.is_set():
                            break
                        payload = self._parse_message(message)
                        if payload is not None:
                            with self._payload_lock:
                                self._latest_payload = payload
            except ConnectionClosed:
                self._connected = False
            except OSError:
                self._connected = False
            finally:
                self._websocket = None
            if not self._stop_event.is_set():
                time.sleep(self.config.reconnect_interval_s)

    def _build_url(self) -> str:
        scheme = "wss" if self.config.use_ssl else "ws"
        path = self.config.path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{scheme}://{self.config.host}:{self.config.port}{path}"

    def _parse_message(self, message: str | bytes) -> dict[str, Any] | None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return None

    def get_action(self) -> dict[str, Any]:
        with self._payload_lock:
            payload = dict(self._latest_payload) if self._latest_payload else None

        if not payload:
            return {}

        action: dict[str, Any] = {}
        for part, path in self.config.pose_paths.items():
            pose = self._resolve_pose(payload, path)
            if not pose:
                continue
            position = self._lookup_pose_value(pose, self.config.position_key, self.config.position_keys)
            orientation = self._lookup_pose_value(pose, self.config.orientation_key, self.config.orientation_keys)
            if position is None or orientation is None:
                continue
            pos_array = np.asarray(position, dtype=float)
            if pos_array.shape != (3,):
                continue
            rot_array = np.asarray(orientation, dtype=float)
            if rot_array.shape != (4,):
                continue
            action[f"{self.config.output_prefix}.{part}.pos"] = pos_array
            action[f"{self.config.output_prefix}.{part}.rot"] = Rotation.from_quat(rot_array)

        if self.config.include_raw:
            action[f"{self.config.output_prefix}.raw"] = payload
        return action

    def _resolve_pose(self, payload: dict[str, Any], path: list[str]) -> dict[str, Any] | None:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        if isinstance(current, dict):
            return current
        return None

    def _lookup_pose_value(self, pose: dict[str, Any], preferred: str, fallbacks: list[str]) -> Any:
        if preferred in pose:
            return pose.get(preferred)
        for key in fallbacks:
            if key in pose:
                return pose.get(key)
        return None

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._websocket is not None:
            try:
                self._websocket.close()
            except Exception:
                pass
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2.0)
        self._connected = False
        self._listener_thread = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
