using ObjectDetector

using Adapt


yolomod = YOLO.v3_608_COCO(batch=10, silent=true, use_gpu=false);

yolomod = YOLO.v3_608_COCO(batch=1, silent=true, use_gpu=false);

batch = emptybatch(yolomod);

@showtime ret=yolomod(batch, detectThresh=0.5, overlapThresh=0.2);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);

using AllocArrays

b = BumperAllocator(2^33); # 8.5 GB

# T = CheckedAllocArray # for testing
T = AllocArray
batch_aa = T(batch);

yolomod_b = deepcopy(yolomod)
yolomod_b.chain = Adapt.adapt(T, yolomod_b.chain)
yolomod_b.W = Dict(k => T(v) for (k,v) in yolomod_b.W)


function bumper_yolomod(b, args...; kwargs...)
    with_allocator(b) do
        _ret = yolomod_b(args...; kwargs...)
        ret = Array(_ret)
        reset!(b)
        return ret
    end
end

println()
ret = @showtime bumper_yolomod(b, batch_aa; detectThresh=0.5, overlapThresh=0.8);

ret = @showtime bumper_yolomod(b, batch_aa; detectThresh=0.5, overlapThresh=0.8);

ret = @showtime bumper_yolomod(b, batch_aa; detectThresh=0.5, overlapThresh=0.8, show_timing=true));
