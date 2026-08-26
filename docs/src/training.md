# [Training](@id training)

```@meta
CurrentModule = ObjectDetector
```

Training is supported for `[yolo]`-output models: both the classic decode (`v3`
family, `v4`, `v4-tiny`) and the `new_coords=1` scaled decode (`v4-csp` and the
rest of the Scaled-YOLOv4 family, the `v7` family). The `yolov2` `[region]`
models are not trainable.

The loss is a modern simplified YOLO loss, CIoU box loss plus binary
cross-entropy objectness and class losses with best-anchor target assignment.
It is not a reimplementation of Darknet's exact loss.

## Batch-norm folding

By default batch-norm is folded into the convolution weights when the weights are
loaded. That makes inference fast and makes training behave like fine-tuning with
frozen batch-norm statistics, which is what you want when adapting pretrained
weights.

For from-scratch training, build the model with `trainable_batchnorm = true` to
keep live, trainable batch-norm layers. Convergence from random initialization
generally needs them.

## Datasets

Datasets follow Darknet conventions: normalized `[class, cx, cy, w, h]` boxes,
with 1-based class indices.

```@docs
TrainSample
load_darknet_dataset
load_darknet_labels
```

## Fine-tuning on a custom dataset

```julia
using ObjectDetector, FileIO, ImageIO

# yolov3-tiny with a fresh 2-class head, trunk from pretrained COCO weights.
# Block 15 is the yolov3-tiny.conv.15 backbone split.
cfg, weights = YOLO.YOLO_MODELS["v3_tiny_COCO"]()
yolomod = YOLO.Yolo(cfg, weights, 1;
    weights_stop_layer = 15,
    cfgchanges = [(:yolo, 1, :classes, 2), (:yolo, 2, :classes, 2),
                  (:convolutional, 10, :filters, 21),
                  (:convolutional, 13, :filters, 21)])
                  # head conv filters = anchors_per_head * (5 + classes)

data = load_darknet_dataset("dataset/images", "dataset/labels")

result = train!(yolomod, data;
    epochs = 50, batchsize = 8, lr = 1e-3,
    image_loader = FileIO.load,
    checkpoint_dir = "checkpoints")

result.losses
```

The model is updated in place, so it can be used for inference straight
afterwards. Truncated Darknet backbone files such as `yolov3-tiny.conv.15` load
directly with `allow_partial_weights = true`, which randomly initializes the rest.

## Training from scratch

```julia
yolomod = YOLO.Yolo(cfg, nothing, 1;
    weights_stop_layer = 0,      # random-init all layers
    trainable_batchnorm = true,  # live batch-norm
    cfgchanges = [...])

train!(yolomod, data;
    epochs = 300, batchsize = 16, lr = 1e-3,
    warmup_batches = 1000,       # linear LR ramp, darknet's burn-in
    flip_augment = true,
    image_loader = FileIO.load)
```

## Splitting off the backbone

Pre-training a backbone on classification, then training the detector from it, is
the usual way to start from something better than noise. `ObjectDetector.backbone`
splits the trunk off as a trainable chain, and `ObjectDetector.copy_backbone!`
puts a trained one back. Neither is exported, because Metalhead.jl exports
`backbone` for the same concept.

```julia
yolo = YOLO.Yolo(cfg, nothing, 1; weights_stop_layer = 0, trainable_batchnorm = true)
trunk = ObjectDetector.backbone(yolo, 15)   # darknet's yolov3-tiny.conv.15 split

# ... train `trunk` on a classification dataset ...

ObjectDetector.copy_backbone!(yolo, cpu(trunk))
save_weights(yolo, "pretrained.weights")
```

[Pre-training a backbone on ImageNet](@ref imagenet-backbone) works through this
end to end.

```@docs
backbone
copy_backbone!
```

## Saving

```@docs
save_weights
```

`save_weights` on a `trainable_batchnorm` model writes true Darknet batch-norm
parameters. Reloading without `trainable_batchnorm` folds them for fastest
inference, with equivalent outputs.

## Reference

```@docs
train!
```
