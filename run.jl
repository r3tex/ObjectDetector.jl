using ObjectDetector

using Functors


yolomod = YOLO.v3_608_COCO(batch=10, silent=true);

yolomod = YOLO.v3_608_COCO(batch=1, silent=true);

batch = emptybatch(yolomod);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);

@showtime yolomod(batch, detectThresh=0.5, overlapThresh=0.8);

using AllocArrays

b = BumperAllocator(2^33); # 8.5 GB

# T = CheckedAllocArray # for testing
T = AllocArray
batch_aa = T(batch);

yolomod_b = deepcopy(yolomod)
function aaize(obj)
    fmap(x -> begin
        x isa AbstractArray || return x
        isbitstype(eltype(x)) || return x
        return T(x)
    end, obj; exclude=x -> x isa AbstractArray{<:Number} || x isa Function)
end

yolomod_b.chain = aaize(yolomod_b.chain)
yolomod_b.W = Dict(k => T(v) for (k,v) in yolomod_b.W)


function bumper_yolomod(b, args...; kwargs...)
    with_allocator(b) do
        _ret = yolomod_b(args...; kwargs...)
        ret = Array(_ret)
        reset!(b)
        return ret
    end
end

ret = @showtime bumper_yolomod(b, batch_aa; detectThresh=0.5, overlapThresh=0.8);

ret = @showtime bumper_yolomod(b, batch_aa; detectThresh=0.5, overlapThresh=0.8);

ret = @showtime bumper_yolomod(b, batch_aa; detectThresh=0.5, overlapThresh=0.8);


y = yolomod_b
y.W[0] = batch_aa

with_allocator(b) do

for i in eachindex(y.chain) # each chain writes to a predefined output

    @time "$i" y.chain[i](y.W[i-1])

end
end
reset!(b)
