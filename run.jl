using ObjectDetector

yolomod = YOLO.v3_608_COCO(batch=10, silent=true, use_gpu=false);

yolomod = YOLO.v3_608_COCO(batch=1, silent=true, use_gpu=false);

batch = emptybatch(yolomod);

@showtime ret=yolomod(batch, detectThresh=0.5, overlapThresh=0.2);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);
println()

using AllocArrays, Adapt

# TODO- move to AllocArrays?
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
    (args...; kw...) -> begin
        with_allocator(b) do
            inputs = Adapt.adapt(T, args)
            ret = Array(model_aa(inputs...; kw...))
            reset!(b)
            return ret
        end
    end
end

# T = CheckedAllocArray # for testing
T = AllocArray

yolomod_aa = wrap_model(yolomod; T)

ret = @showtime yolomod_aa(batch; detectThresh=0.5, overlapThresh=0.8);

ret = @showtime yolomod_aa(batch; detectThresh=0.5, overlapThresh=0.8);

ret = @showtime yolomod_aa(batch; detectThresh=0.5, overlapThresh=0.8, show_timing=true);
