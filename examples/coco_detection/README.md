# Detection training on COCO

Trains a YOLO detector on COCO with the package's `train!`, reading the official
annotations rather than a darknet-converted copy of them.

```
julia --project -e 'using Pkg; Pkg.instantiate()'

# val2017 is 5k images and ~1 GB, enough to exercise the whole pipeline.
# Swap in train2017 (118k images, ~18 GB) for a real run.
mkdir -p coco && cd coco
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip && cd ..

julia --project -t auto train.jl --data coco --epochs 5
```

| File | |
|:--|:--|
| `coco.jl` | COCO instances JSON to `TrainSample`, with the class-id remapping |
| `train.jl` | model setup and the call to `train!` |

Options: `--data --split --out --cfg --from --weights --classes --epochs
--batchsize --lr --res --limit --warmup --backend`.

`--from` picks what the weights start as:

- `pretrained` (default) fine-tunes the COCO-pretrained detector.
- `backbone` starts from an ImageNet-pre-trained trunk, which is what
  [`../imagenet_backbone`](../imagenet_backbone) produces. Needs `--weights`.
- `scratch` random-initializes everything, which needs far more data and epochs
  than this example is set up for.

`--classes person,car` cuts the model down to a subset, rebuilding the `[yolo]`
layers and the convolutions feeding them for the smaller class count. That is the
common real use of this: adapting a pretrained detector to your own labels.
