# ImageNet pre-training for a YOLO backbone

Trains the convolutional trunk of an ObjectDetector YOLO model as an image
classifier with [ImageNetDataset.jl](https://github.com/Julia-XAI/ImageNetDataset.jl),
then writes it back out as darknet `.weights` for `train!` to fine-tune into a
detector.

ImageNet has no bounding boxes, so this trains the trunk only, which is the role
darknet's `yolov3-tiny.conv.15` plays. See the
[tutorial](../../docs/src/tutorials/imagenet_backbone.md) for the full write-up.

```
julia --project -e 'using Pkg; Pkg.instantiate()'

# 10-class real-ImageNet subset, no registration needed
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz

julia --project -t auto train.jl --data imagenette2-320 --epochs 5
```

| File | |
|:--|:--|
| `backbone.jl` | pulls the trunk out of a `YOLO.Yolo` and wraps it in a classifier |
| `data.jl` | ImageNetDataset loaders and threaded batch assembly |
| `train.jl` | training loop, checkpointing, weight export |
| `bench.jl` | per-step throughput across backends and batch sizes |

Options: `--data --out --cfg --backend --epochs --batchsize --lr --res --nconv --full`.
