import os
import sys
sys.path.append(os.getcwd())        # for relative imports in root directory

weight_root = "Depth_FCN_2/model_res"

try:
    from Depth_FCN_2.FCN import DepthBasedFCN

    model = DepthBasedFCN(3)

    import torch
    
    model.load_state_dict(torch.load(os.path.join(weight_root,"best_weights.pt"),
                                    map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    torch.manual_seed(1)

    torch_input = torch.randn(1, 3, 113, 102)

    output = model(torch_input)
    print(output.shape)

    torch.onnx.export(model,         # model being run 
            torch_input,       # model input (or a tuple for multiple inputs) 
            os.path.join(weight_root,"best_model.onnx"),       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=10,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput'], # the model's output names 
            dynamic_axes={'modelInput' : {0 : 'batch_size', 2 : 'input_x', 3 : 'input_y'},    # variable length axes
                          'modelOutput' : {0 : 'batch_size', 2 : 'output_x', 3 : 'output_y'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

    print("DONE SAVING")

except Exception as e:
    print(e)
    sys.exit()
