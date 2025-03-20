struct WrappedModel
    model::AbstractModel
    allocator
    T
    model_aa
end
function (wm::WrappedModel)(args...; kw...)
    with_allocator(wm.allocator) do
        inputs = Adapt.adapt(wm.T, args)
        ret = Array(wm.model_aa(inputs...; kw...))
        reset!(wm.allocator)
        return ret
    end
end
emptybatch(wm::WrappedModel) = emptybatch(wm.model)
prepareImage(img::AbstractArray, model::WrappedModel) = prepareImage(img, model.model)
uses_gpu(wm::WrappedModel) = uses_gpu(wm.model)
drawBoxes(img, wm::WrappedModel, padding, results; kw...) = drawBoxes(img, wm.model, padding, results; kw...)

"""
    wrap_model(model; n_bytes=2^33, T=AllocArray)

Wraps a model to use a bump allocator for temporary arrays. The `n_bytes` of memory will be preallocated upfront once when `wrap_model` is called, and re-used for every batch inferred over. The type `T` can be set to `CheckedAllocArray` for testing purposes.

## Example

```julia
yolomod = YOLO.v3_608_COCO(batch=1, silent=true, use_gpu=false);
batch = emptybatch(yolomod);
@time yolomod(batch) #  0.900718 seconds (6.89 k allocations: 6.295 GiB, 5.53% gc time)

# Now let's use our bump allocator
yolomod_aa = wrap_model(yolomod; T=AllocArray)
@time yolomod_aa(batch) #  0.857606 seconds (8.74 k allocations: 697.273 KiB)

```
"""
function wrap_model(model; n_bytes=2^33, T=AllocArray)
    b = BumperAllocator(n_bytes)
    model_aa = Adapt.adapt(T, model)
    return WrappedModel(model, b, T, model_aa)
end
