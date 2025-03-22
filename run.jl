using AllocArrays
using AllocArrays.Bumper
using ObjectDetector

kw = (; use_gpu=false, select=[1, 3, 6], verbose=false)

println("No bumper:")
ObjectDetector.benchmark(; disallow_bumper=true, kw...)

N = 2^33
println("Using `AllocBuffer($N)` (preallocate $(Base.format_bytes(N))):")
ObjectDetector.benchmark(; allocator=BumperAllocator(AllocBuffer(N)), kw...)

for N in [2^20, 2^25, 2^26, 2^27, 2^28, 2^29, 2^30, 2^31, 2^32]
    println("Using `SlabBuffer{$N}()` (slabs of size $(Base.format_bytes(N))):")
    ObjectDetector.benchmark(; allocator=BumperAllocator(SlabBuffer{N}()), kw...)
end
