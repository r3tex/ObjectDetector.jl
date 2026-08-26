using Documenter
using ObjectDetector

makedocs(;
    sitename = "ObjectDetector.jl",
    authors = "Robert Luciani, Ian Butterworth and contributors",
    modules = [ObjectDetector],
    # The package documents plenty of internals; only require that the exported
    # API is covered by a @docs block.
    checkdocs = :exports,
    format = Documenter.HTML(;
        canonical = "https://r3tex.github.io/ObjectDetector.jl",
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Pretrained models" => "models.md",
        "Training" => "training.md",
        "Tutorials" => [
            "Pre-training a backbone on ImageNet" => "tutorials/imagenet_backbone.md",
            "Detection training on COCO" => "tutorials/coco_detection.md",
        ],
        "Acceleration" => "acceleration.md",
        "API reference" => "api.md",
    ],
)

deploydocs(; repo = "github.com/r3tex/ObjectDetector.jl.git", push_preview = true)
