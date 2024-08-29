import os
import sys
sys.path.append(os.getcwd())        # for relative imports in root directory

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

weight_root = "Depth_FCN_2/model_res"

try:
    modelPath = os.path.join(weight_root,"best_model.onnx")

    import onnx

    onnx_model = onnx.load(modelPath)
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(modelPath, providers=["CPUExecutionProvider"])

    import torch

    torch.manual_seed(1)
    torch_input = torch.randn(1, 3, 64, 64)
    print(to_numpy(torch_input).shape)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])

except Exception as e:
    print(e)
    sys.exit()
