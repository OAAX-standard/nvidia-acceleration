#!/usr/bin/env python3
"""
Generate a tiny ONNX identity model and bundle it for conversion.

Output: tests/artifacts/bundle.zip  (model.onnx + config.json)

The model has one input ("input", float32, shape [1,4]) and returns it unchanged.
Small and fast to convert — good for CI / smoke tests.
"""
import json
import os
import zipfile

import onnx
from onnx import TensorProto, helper

SHAPE = [1, 4]

def make_model():
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "identity-test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7  # keep compatible with older ORT (max IR 12)
    onnx.checker.check_model(model)
    return model

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    model_path  = os.path.join(out_dir, "model.onnx")
    config_path = os.path.join(out_dir, "config.json")
    bundle_path = os.path.join(out_dir, "bundle.zip")

    onnx.save(make_model(), model_path)
    print(f"Saved {model_path}")

    with open(config_path, "w") as f:
        json.dump({"precision": "fp16", "workspace_gb": 1}, f)
    print(f"Saved {config_path}")

    with zipfile.ZipFile(bundle_path, "w") as z:
        z.write(model_path, "model.onnx")
        z.write(config_path, "config.json")
    print(f"Saved {bundle_path}")
