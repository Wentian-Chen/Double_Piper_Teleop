import types
import sys

import numpy as np


class FakeSMOLVLA:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instruction = None
        self.last_observation = None

    def random_set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, images, state):
        self.last_observation = {"images": images, "state": state}

    def get_action_chunk(self):
        return np.array([1, 2, 3, 4, 5, 6, 0.5], dtype=np.float32)


def test_smolvla_model_predict_with_alias_keys(monkeypatch):
    from vla_infer.models.smolvla_model import SmolVLAModel

    fake_module = types.ModuleType("policy.smolvla.inference_model")
    fake_module.SMOLVLA = FakeSMOLVLA
    monkeypatch.setitem(sys.modules, "policy.smolvla.inference_model", fake_module)

    model = SmolVLAModel(model_path="dummy", device="cpu")
    obs = {
        "cmd": "pick banana",
        "front_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
        "state": np.arange(7, dtype=np.float32),
    }

    result = model.predict(obs)

    assert "action" in result
    assert result["action"].shape == (1, 7)
    assert np.allclose(result["action"][0, :6], np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))


def test_vla_adapter_model_predict_and_state_padding(monkeypatch):
    from vla_infer.models.vla_adapter_model import VLAAdapterModel

    class FakeTorch:
        class _Cuda:
            @staticmethod
            def is_available():
                return False

        cuda = _Cuda()

        @staticmethod
        def device(name):
            return name

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.device = None

        def eval(self):
            self.eval_called = True

        def to(self, device):
            self.device = device
            return self

    captured = {"input_obs": None}

    class FakeEvalModule:
        class EvalConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        @staticmethod
        def initialize_model(cfg):
            return FakeModel(), "action_head", "proprio_projector", None, "processor"

        @staticmethod
        def get_vla_action(cfg, model, processor, input_obs, task, action_head=None, proprio_projector=None, use_minivlm=True):
            captured["input_obs"] = input_obs
            return np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]], dtype=np.float32)

    monkeypatch.setattr(
        "vla_infer.models.vla_adapter_model.importlib.import_module",
        lambda name: FakeTorch(),
    )
    monkeypatch.setattr(VLAAdapterModel, "_load_eval_module", staticmethod(lambda _: FakeEvalModule))

    model = VLAAdapterModel(model_path="dummy", device="cpu")
    obs = {
        "cmd": "pick banana",
        "cam_head": np.zeros((8, 8, 3), dtype=np.uint8),
        "cam_wrist": np.ones((8, 8, 3), dtype=np.uint8),
        "state": np.arange(7, dtype=np.float32),
    }

    result = model.predict(obs)

    assert "action" in result
    assert result["action"].shape == (1, 7)
    assert captured["input_obs"] is not None
    assert captured["input_obs"]["state"].shape == (8,)
