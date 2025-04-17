module CUDAExt

using CUDA
import Flux
import ObjectDetector.YOLO: maxpool, clipdetect!, findmax!, keepdetections, extend_for_attributes, upsample

const CU_FUNCTIONAL = Ref{Bool}()

function cu_functional()
    if !isassigned(CU_FUNCTIONAL)
        CUDA.allowscalar(false)
        CU_FUNCTIONAL[] = CUDA.functional()
    end
    return CU_FUNCTIONAL[]
end


function clipdetect!(input::CuArray, conf)
    rows, cols = size(input)
    @cuda blocks=cols threads=1024 kern_clipdetect(input, conf)
end
function kern_clipdetect(input::CuDeviceArray, conf::Float32)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cols = gridDim().x
    if idx <= cols
        @inbounds input[end-2, idx] = ifelse(input[end-2, idx] > conf, input[end-2, idx], Float32(0.0))
    end
    return
end


function findmax!(input::CuArray)
    rows, cols = size(input)
    idst, idend = 6, rows - 3
    @cuda blocks=cols threads=rows kern_findmax!(input, idst, idend)
end
function kern_findmax!(input::CuDeviceMatrix{T}, idst::Integer, idend::Integer) where {T}
    if threadIdx().x == idend
        j = blockIdx().x
        val = zero(T)
        idx = zero(T)
        for i in idst:idend
            if input[i, j] > val
                val = input[i, j]
                idx = i
            end
        end
        @inbounds input[end-2, j] = val
        @inbounds input[end-1, j] = idx - idst + 1
    end
    return
end

function maxpool(x::CuArray; siz=2, stride=1, pad)
    if siz == 2 && stride == 1 && pad == (0, 1, 0, 1)
        # cuDNN doesn't support asymmetric padding
        xp = CUDA.zeros(Float32, size(x,1)+1, size(x,2)+1, size(x,3), size(x,4))
        xp[1:size(x,1), 1:size(x,2), :, :] .= x
        xp[end, :, :, :] .= xp[end-1, :, :, :] # replicate last row
        xp[:, end, :, :] .= xp[:, end-1, :, :] # replicate last column
        pdims = Flux.PoolDims(xp, (siz, siz); stride=(stride, stride), padding=(0, 0, 0, 0)) # Use padding=0
        return Flux.maxpool(xp, pdims)
    else
        # For all other cases (symmetric padding or stride != 1), padding is handled directly by Flux.PoolDims
        pdims = Flux.PoolDims(x, (siz, siz); stride=(stride, stride), padding=pad)
        return Flux.maxpool(x, pdims)
    end
end

function keepdetections(input::CuArray) # THREADS:BLOCKS CAN BE OPTIMIZED WITH BETTER KERNEL
    rows, cols = size(input)
    bools = CUDA.zeros(Int32, cols)
    @cuda blocks=cols threads=rows kern_genbools(input, bools)
    idxs = cumsum(bools)
    n = count(isone, bools)
    output = CuArray{Float32, 2}(undef, rows, n)
    @cuda blocks=cols threads=rows kern_keepdetections(input, output, bools, idxs)
    return output
end
function kern_genbools(input::CuDeviceArray, output::CuDeviceArray)
    col = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cols = gridDim().x
    if col < cols && input[end-2, col] > Float32(0)
        @inbounds output[col] = Int32(1)
    end
    return
end
@inline function kern_keepdetections(input::CuDeviceArray, output::CuDeviceArray,
    bools::CuDeviceArray, idxs::CuDeviceArray)
    col = blockIdx().x
    row = threadIdx().x
    if bools[col] == Int32(1)
        idx = idxs[col]
        @inbounds output[row, idx] = input[row, col]
    end
    return
end

function extend_for_attributes(weights::CuArray, w, h, bo, ba)
    return cat(weights, CUDA.zeros(Float32, w, h, 4, bo, ba), dims = 3)
end

"""
    upsample(a, stride)

Optimized upsampling without indexing for better GPU performance
"""
function upsample(a::CuArray, stride)
    m1, n1, o1, p1 = size(a)
    ar = reshape(a, (1, m1, 1, n1, o1, p1))
    b = CUDA.ones(Float32, stride, 1, stride, 1, 1, 1)
    return reshape(ar .* b, (m1 * stride, n1 * stride, o1, p1))
end


end # module