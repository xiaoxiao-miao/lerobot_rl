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

from dataclasses import field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xr")
class XRTeleopConfig(TeleoperatorConfig):
    """Configuration for XR teleoperation devices."""

    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/"
    use_ssl: bool = False
    reconnect_interval_s: float = 1.0
    position_key: str = "position"
    orientation_key: str = "orientation"
    position_keys: list[str] = field(default_factory=lambda: ["position", "pos"])
    orientation_keys: list[str] = field(default_factory=lambda: ["orientation", "quat", "rotation", "rot"])
    output_prefix: str = "xr"
    include_raw: bool = False
    pose_paths: dict[str, list[str]] = field(
        default_factory=lambda: {
            "head": ["head"],
            "left_wrist": ["left_wrist"],
            "right_wrist": ["right_wrist"],
            "left_hand": ["left_hand"],
            "right_hand": ["right_hand"],
            "left_controller": ["left_controller"],
            "right_controller": ["right_controller"],
        }
    )
