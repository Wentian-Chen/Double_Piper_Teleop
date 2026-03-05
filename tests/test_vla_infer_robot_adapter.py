import numpy as np
import pytest

from vla_infer.robots.piper_single import PiperSingleRobot


class FakePiperSingle:
    def __init__(self):
        self.setup_called = False
        self.reset_called = False
        self.last_move_data = None

    def set_up(self):
        self.setup_called = True

    def reset(self):
        self.reset_called = True

    def get(self):
        return {
            "arm": {
                "left_arm": {
                    "joint": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "qpos": 0.0,
                    "gripper": 0.2,
                }
            },
            "image": {
                "cam_head": {
                    "color": np.zeros((16, 16, 3), dtype=np.uint8),
                    "depth": np.zeros((16, 16), dtype=np.uint16),
                },
                "cam_wrist": {
                    "color": np.ones((16, 16, 3), dtype=np.uint8),
                    "depth": np.ones((16, 16), dtype=np.uint16),
                },
            },
        }

    def move(self, move_data):
        self.last_move_data = move_data


class FakePiperSingleListFormat(FakePiperSingle):
    def get(self):
        return [
            {
                "left_arm": {
                    "joint": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "qpos": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                    "gripper": 0.7,
                }
            },
            {
                "cam_head": {"color": np.zeros((8, 8, 3), dtype=np.uint8)},
                "cam_wrist": {"color": np.ones((8, 8, 3), dtype=np.uint8)},
            },
        ]


def test_get_observation_supports_dict_format():
    robot = PiperSingleRobot(auto_setup=False, robot_cls=FakePiperSingle)

    obs = robot.get_observation()

    assert obs["state"].shape == (7,)
    assert np.allclose(obs["joint"], np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    assert obs["qpos"].shape == (6,)
    assert np.allclose(obs["qpos"], np.zeros(6, dtype=np.float32))
    assert obs["cam_head"].shape == (16, 16, 3)
    assert obs["cam_wrist"].shape == (16, 16, 3)


def test_get_observation_supports_list_format():
    robot = PiperSingleRobot(auto_setup=False, robot_cls=FakePiperSingleListFormat)

    obs = robot.get_observation()

    assert obs["state"].shape == (7,)
    assert np.allclose(obs["qpos"], np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32))


def test_apply_action_sends_left_arm_joint_and_gripper():
    robot = PiperSingleRobot(auto_setup=False, robot_cls=FakePiperSingle)

    action = np.array([[1, 2, 3, 4, 5, 6, 0.9]], dtype=np.float32)
    robot.apply_action({"action": action})

    move_data = robot._robot.last_move_data
    assert "arm" in move_data
    assert "left_arm" in move_data["arm"]
    assert move_data["arm"]["left_arm"]["joint"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert move_data["arm"]["left_arm"]["gripper"] == pytest.approx(0.9)
