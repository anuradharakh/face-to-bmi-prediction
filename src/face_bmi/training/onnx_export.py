import torch


class MultiTaskONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs["bmi"], outputs["gender_logits"]


def export_bmi_model_to_onnx(model, onnx_path, device):
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["bmi_output"],
        external_data=True,
    )


def export_multitask_model_to_onnx(model, onnx_path, device):
    model.eval()

    wrapped_model = MultiTaskONNXWrapper(model).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["bmi_output", "gender_logits"],
        external_data=True,
    )