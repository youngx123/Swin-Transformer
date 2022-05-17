
__torch.roll导出onnx报错及解决__

pytorch version : 1.7

>https://github.com/pytorch/pytorch/issues/56355
![](https://github.com/youngx123/pic/blob/main/torch.roll.png?raw=true)

模型转化测试
```python
class M(torch.nn.Module):
    def __init__(self, shifts, dims):
        super(M, self).__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x):
        return torch.roll(x, self.shifts, self.dims)


net = M(-1, 1)
img = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(3, 3)
print(img)
out = net(img)
print(out.shape, "\n\n", out)
torch.onnx.export(net, img, "swin.onnx", verbose=0, training=torch.onnx.TrainingMode.EVAL,
                    input_names=["input"], output_names=["outnode"], opset_version=11)

import onnxruntime
import onnx

onxmodel = onnxruntime.InferenceSession("swin.onnx")
img_np = img.numpy()
out2 = onxmodel.run(None, {"input":img_np})
print("#####################")
print(out2[0])
```
implement Swin transformer

仅供自己学习Swin transformer网络结构和网络内部尺度变换以及处理流程


项目参考：
>https://github.com/microsoft/Swin-Transformer

padding操作可以进行任意尺寸图像输入
>https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py
