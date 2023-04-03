from collections import deque
from enum import Enum

import carla
import numpy as np
import torch
import yaml

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from mile.utils.geometry_utils import calculate_geometry, gps_to_location, vec_global_to_ref
from mile.trainer import WorldModelTrainer
from mile.config import get_cfg
from mile.constants import CARLA_FPS


def get_entry_point():
    return "MILEAgent"


class RouteCommand(Enum):
    VOID = -1
    # VOID = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class MILEAgent(AutonomousAgent):
    def sensors(self):
        return [
            {"type": "sensor.speedometer", "id": "speed"},
            {"type": "sensor.camera.rgb", "id": "central_rgb", **self._camera_parameters},
            {"type": "sensor.other.gnss", "id": "gps", "x": 0.0, "y": 0.0, "z": 0.0},
            {
                "type": "sensor.other.imu",
                "id": "imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        ]

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        _cfg = get_cfg()
        _cfg.merge_from_file(path_to_conf_file)
        self._training_module = WorldModelTrainer.load_from_checkpoint(_cfg.PRETRAINED.PATH, path_to_conf_file=path_to_conf_file)
        self._policy = self._training_module.to("cuda").float()
        self._policy = self._policy.eval()
        self._cfg = self._training_module.cfg

        self._camera_parameters = {
            # Coordinate frame (Unreal): X-front, Y-right, Z-up
            "x": self._cfg.IMAGE.CAMERA_POSITION[0],
            "y": self._cfg.IMAGE.CAMERA_POSITION[1],
            "z": self._cfg.IMAGE.CAMERA_POSITION[2],
            "roll": self._cfg.IMAGE.CAMERA_ROTATION[0],
            "pitch": self._cfg.IMAGE.CAMERA_ROTATION[0],
            "yaw": self._cfg.IMAGE.CAMERA_ROTATION[0],
            "width": self._cfg.IMAGE.SIZE[1],
            "height": self._cfg.IMAGE.SIZE[0],
            "fov": self._cfg.IMAGE.FOV,
        }
        n_image_per_stride = int(CARLA_FPS * self._cfg.DATASET.STRIDE_SEC)
        self._input_queue_size = (self._cfg.RECEPTIVE_FIELD - 1) * n_image_per_stride + 1
        self._sequence_indices = range(0, self._input_queue_size, n_image_per_stride)
        self._input_queue = {
            "image": deque(maxlen=self._input_queue_size),
            "speed": deque(maxlen=self._input_queue_size),
            "intrinsics": deque(maxlen=self._input_queue_size),
            "extrinsics": deque(maxlen=self._input_queue_size),
            "route_command": deque(maxlen=self._input_queue_size),
            "gps_vector": deque(maxlen=self._input_queue_size),
            "action": deque(maxlen=self._input_queue_size),
        }
        self._idx_plan = -1
        self._vehicle_control = None
        self._skip = False

    def destroy(self):
        for v in self._input_queue.values():
            v.clear()

    def run_step(self, input_data, timestamp):
        # Leaderboard is at 20fps, skip one frame every 2 frames to match how the model was trained
        if not self._skip:
            with torch.no_grad():
                self._extract_data(input_data)
                model_input = self._get_model_input()
                model_output = self._policy(model_input, deployment=True)
                self._vehicle_control = self._process_output(model_output)

                self._visualise_outputs(model_input, model_output, timestamp)

        self._skip = not self._skip
        return self._vehicle_control

    def _extract_data(self, input_data):
        # Extract data from Carla dict
        speed_float = input_data["speed"][1]["speed"]
        image_np = input_data["central_rgb"][1].transpose((2, 0, 1))  # Transpose H W C to C H W
        image_np = image_np[0:3]  # Drop alpha channel from BGR-A
        image_np = image_np[::-1]  # BGR -> RGB
        intrinsics_np, extrinsics_np = calculate_geometry(
            self._camera_parameters["fov"],
            self._camera_parameters["height"],
            self._camera_parameters["width"],
            self._camera_parameters["x"],
            self._camera_parameters["y"],
            self._camera_parameters["z"],
            self._camera_parameters["roll"],
            self._camera_parameters["pitch"],
            self._camera_parameters["yaw"],
        )
        gps_vector, route_command = self._extract_current_navigation(input_data)

        # Move to tensors to GPU and update queue
        image_torch = torch.from_numpy(image_np.copy()).cuda()
        intrinsics_torch = torch.from_numpy(intrinsics_np).cuda()
        extrinsics_torch = torch.from_numpy(extrinsics_np).cuda()
        speed_torch = torch.tensor(speed_float).unsqueeze(0).float().cuda()
        gps_vector_torch = torch.from_numpy(gps_vector).cuda()
        route_command_torch = torch.from_numpy(route_command).cuda()
        self._update_queue(
            image_torch,
            intrinsics_torch,
            extrinsics_torch,
            speed_torch,
            gps_vector_torch,
            route_command_torch,
        )

    def _extract_current_navigation(self, input_data):
        # Select current index from plan
        next_gps, _ = self._global_plan[self._idx_plan + 1]
        ego_gps = input_data["gps"][1]  # Latitude, Longitude, Altitude
        next_vec_in_global = gps_to_location(next_gps.values()) - gps_to_location(ego_gps)
        # IMU data: Acc[x,y,z], AngVel[x,y,z], Orientation (radians with north (0.0, -1.0, 0.0) in UE)
        orientation = input_data["imu"][1][-1]
        compass = 0.0 if np.isnan(orientation) else orientation
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = vec_global_to_ref(next_vec_in_global, ref_rot_in_global)
        if np.sqrt(loc_in_ev.x**2 + loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
            self._idx_plan += 1
        self._idx_plan = min(self._idx_plan, len(self._global_plan) - 2)

        # Select route command and relevant GPS point using relevant index
        _, route_command_0 = self._global_plan[max(0, self._idx_plan)]
        gps_point, route_command_1 = self._global_plan[self._idx_plan + 1]
        if (route_command_0 in [RouteCommand.CHANGELANELEFT, RouteCommand.CHANGELANERIGHT]) and (
            route_command_1 not in [RouteCommand.CHANGELANELEFT, RouteCommand.CHANGELANERIGHT]
        ):
            route_command = route_command_1
        else:
            route_command = route_command_0
        route_command = (
            RouteCommand.LANEFOLLOW if route_command == RouteCommand.VOID else route_command
        )

        # Process route command to start at 0
        route_command = route_command.value - 1
        route_command = np.array(route_command, dtype=np.int64)

        # Get vector to target gps
        target_vec_in_global = gps_to_location(gps_point.values()) - gps_to_location(ego_gps)
        loc_in_ev = vec_global_to_ref(target_vec_in_global, ref_rot_in_global)
        gps_vector = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)

        return gps_vector, route_command

    def _update_queue(
        self, image, intrinsics, extrinsics, speed, gps_vector_torch, route_command_torch
    ):
        # Add to queue
        self._input_queue["image"].append(image)
        self._input_queue["intrinsics"].append(intrinsics)
        self._input_queue["extrinsics"].append(extrinsics)
        self._input_queue["speed"].append(speed)
        self._input_queue["route_command"].append(route_command_torch)
        self._input_queue["gps_vector"].append(gps_vector_torch)

        # When the action buffer is empty (first forward pass), add a dummy zero action
        if len(self._input_queue["action"]) == 0:
            self._input_queue["action"].append(torch.zeros(2, device=torch.device("cuda")))

        # When the buffers are not full (first forward pass), replicate the last inputs to fill it up
        for key in self._input_queue.keys():
            while len(self._input_queue[key]) < self._input_queue_size:
                self._input_queue[key].append(self._input_queue[key][-1])

    def _get_model_input(self):
        # Prepare model input by selecting from the input queue and adding batch and sequence dimensions
        model_input = {
            key: torch.stack(list(val), axis=0).unsqueeze(0).clone()
            for key, val in self._input_queue.items()
        }
        model_input["action"] = torch.cat(
            [model_input["action"][:, 1:], torch.zeros_like(model_input["action"][:, -1:])], dim=1
        )

        # Select the correct elements in the model input according to the sequence indices required by the model
        for k, v in model_input.items():
            model_input[k] = v[:, self._sequence_indices]

        return model_input

    def _process_output(self, model_output):
        # Append action to queue
        actions = (
            torch.cat([model_output["throttle_brake"], model_output["steering"]], dim=-1)[0, 0]
            .cpu()
            .detach()
            .numpy()
        )
        self._input_queue["action"].append(torch.from_numpy(actions).cuda())

        acceleration = model_output["throttle_brake"].item()
        steering = model_output["steering"].item()
        if acceleration >= 0.0:
            throttle = acceleration
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acceleration)

        # Ensure sane values
        throttle = np.clip(throttle, 0, 1)
        steering = np.clip(steering, -1, 1)
        brake = np.clip(brake, 0, 1)

        return carla.VehicleControl(throttle=throttle, steer=steering, brake=brake)

    def _visualise_outputs(self, model_input, model_output, timestamp):

        # Very temporary, will clean up.
        save_path = './vis_tmp'
        image = model_input['image'][0, -1].cpu().numpy().transpose((1, 2, 0))

        # unnormalise image
        img_mean = np.array(self._cfg.IMAGE.IMAGENET_MEAN)
        img_std = np.array(self._cfg.IMAGE.IMAGENET_STD)
        image = (255 * (image * img_std + img_mean)).astype(np.uint8)
        import os
        from PIL import Image
        image = Image.fromarray(image)
        image.save(os.path.join(save_path, f'image_{int(timestamp * 10):06d}.png'))
