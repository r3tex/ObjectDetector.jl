using CUDAnative
using CuArrays

function kern_clipdetect(input::CuDeviceArray, conf::Float32)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cols = gridDim().x
    if idx < cols
        @inbounds input[5, idx] = ifelse(input[5, idx] > conf, input[5, idx], Float32(0.0))
    end
    return
end

function kern_genbools(input::CuDeviceArray, output::CuDeviceArray)
    col = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cols = gridDim().x
    if col < cols && input[5, col] > Float32(0)
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


