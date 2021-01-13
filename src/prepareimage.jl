using ImageFiltering
using ImageTransformations
using ImageCore

"""
    resizekern(source_size::Tuple{Int,Int}, dest_size::Tuple{Int,Int})

Create an image resize blur kernel, for use before reducing image size to avoid aliasing.
"""
function resizekern(source_size, dest_size)
    source_size[1:2] == dest_size[1:2] && return nothing
    # Blur before resizing to prevent aliasing (kernel size dependent on both source and target image size)
    σ = 0.1 .* (source_size[1:2] ./ dest_size[1:2])

    if first(σ) < 1
        return ImageFiltering.KernelFactors.gaussian(σ)
    else
        return ImageFiltering.KernelFactors.IIRGaussian(σ)
    end
end

"""
    sizethatfits(original_size::Tuple{Int,Int}, target_shape::Tuple{Int,Int})

Takes an original image size, and fits it to a target shape for image resizing. Maintains aspect ratio.
"""
function sizethatfits(src_size, dst_size)
    scale = minimum([dst_size[1]/src_size[1], dst_size[2]/src_size[2]])
    result = floor.(Int, src_size[[1,2]] .* scale)
    return result
end

"""
    getpadding(size_inner, size_outer)

Get padding in pixels, with size_inner placed at the center of size_outer.
size_inner must be equal or smaller than size_outer
"""
function getpadding(size_inner, size_outer)
    size_diff = size_outer[1:2] .- size_inner[1:2]
    @assert all(size_diff .>= 0) "`size_inner` should be equal or smaller in both dimensions than `size_outer`"
    padding = [floor(Int, size_diff[1]/2), floor(Int, size_diff[2]/2), ceil(Int, size_diff[1]/2), ceil(Int, size_diff[2]/2)]
end

"""
    prepareImage(img::AbstractArray{T}, model::U) where {T<:ImageCore.Colorant, U<:AbstractModel}
    prepareImage(img::AbstractArray{T}, img_resized_size::Tuple, kern) where {T<:ImageCore.Colorant}
    prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:Real}
    prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:ImageCore.Colorant}

Loads and prepares (resizes + pads) an image to fit within a given shape.
Input images should be column-major (julia default), and will be converted to row-major (darknet).
"""
function prepareImage(img::AbstractArray{T}, model::AbstractModel) where {T<:ImageCore.Colorant}
    modelInputSize = getModelInputSize(model)
    if ndims(img) == 3 && size(img)[[3,2,1]] == modelInputSize[1:3]
        return (gpu(PermutedDimsArray(Float32.(channelview(img)), [3,2,1])), [0,0,0,0])
    elseif size(img)[[2,1]] == modelInputSize[1:2] && modelInputSize[3] == 1
        return (gpu(Float32.(reshape(PermutedDimsArray(channelview(img), [2,1]), modelInputSize...))), [0,0,0,0])
    elseif size(img)[[2,1]] == modelInputSize[1:2] && modelInputSize[3] == 3
        img_chv = channelview(img)
        if ndims(img_chv) == 2
            return (gpu(Float32.(repeat(reshape(PermutedDimsArray(img_chv, [2,1]), size(img,2), size(img,1), 1), outer=[1,1,modelInputSize[3]]))), [0,0,0,0])
        elseif size(img_chv, 1) == 1
            return (gpu(Float32.(repeat(PermutedDimsArray(img_chv, [3,2,1]), outer=[1,1,modelInputSize[3]]))), [0,0,0,0])
        elseif size(img_chv, 1) == 3
            return (gpu(Float32.(PermutedDimsArray(img_chv, [3,2,1]))), [0,0,0,0])
        elseif size(img_chv, 1) == 4
            return (gpu(Float32.(PermutedDimsArray(view(img_chv,1:3,:,:), [3,2,1]))), [0,0,0,0])

        else
            error("Image element type $(eltype(img)) not supported")
        end
    else
        img_size = size(img)[[2,1]]
        img_resized_size = sizethatfits(img_size, modelInputSize)
        kern = resizekern(img_size, img_resized_size)
        return prepareImage(img, modelInputSize, kern)
    end
end
function prepareImage(img::AbstractArray{Float32}, model::AbstractModel)
    modelInputSize = getModelInputSize(model)
    if ndims(img) == 3 && size(img) == modelInputSize[1:3]
        return (gpu(img), [0,0,0,0])
    elseif ndims(img) == 3 && size(img)[[3,2,1]] == modelInputSize[1:3]
        return (gpu(PermutedDimsArray(img, [3,2,1])), [0,0,0,0])
    elseif size(img)[[2,1]] == modelInputSize[1:2] && modelInputSize[3] == 1
        return (gpu(Float32.(reshape(PermutedDimsArray(img, [2,1]), modelInputSize...))), [0,0,0,0])
    elseif ndims(img) == 2 && size(img)[[2,1]] == modelInputSize[1:2] && modelInputSize[3] == 3
        return (gpu(Float32.(repeat(reshape(PermutedDimsArray(img, [2,1]), size(img,2), size(img,1), 1), outer=[1,1,modelInputSize[3]]))), [0,0,0,0])
    else
        img_size = size(img)[[2,1]]
        img_resized_size = sizethatfits(img_size, modelInputSize)
        kern = resizekern(img_size, img_resized_size)
        return prepareImage(img, modelInputSize, kern)
    end
end

function prepareImage(img::AbstractArray{Float32}, modelInputSize::Tuple)
    img_size = size(img)[[2,1]]
    img_resized_size = sizethatfits(img_size, modelInputSize)
    kern = resizekern(img_size, img_resized_size)
    return prepareImage!(gpu(zeros(Float32, modelInputSize[1:3])), img, kern)
end

prepareImage(img::AbstractArray{Float32}, modelInputSize::Tuple, kern) =
    prepareImage!(gpu(zeros(Float32, modelInputSize[1:3])), img, kern)

prepareImage(img::AbstractArray{T}, modelInputSize::Tuple, kern) where {T<:ImageCore.Colorant} =
    prepareImage!(gpu(zeros(Float32, modelInputSize[1:3])), img, kern)

function prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{Float32}, kern)
    #TODO: Make this multiple-dispatchy
    if ndims(img) == 3 && size(img) == size(dest_arr)
        return (gpu(img), [0,0,0,0])
    elseif ndims(img) == 3 && size(img)[[2,1,3]] == size(dest_arr)
        return (gpu(PermutedDimsArray(img, [2,1,3])), [0,0,0,0])
    elseif ndims(img) == 2 && size(img)[[2,1]] == size(dest_arr)[1:2]
        return (gpu(reshape(PermutedDimsArray(img, [2,1]), size(img,2), size(img, 1), 1)))
    elseif ndims(img) == 2
        return prepareImage!(dest_arr, colorview(Gray, img), kern)
    elseif size(img, 1) == 1
        return prepareImage!(dest_arr, colorview(Gray, img)[1,:,:], kern)
    elseif size(img, 3) == 1
        return prepareImage!(dest_arr, colorview(Gray, img)[:,:,1], kern)
    elseif size(img, 3) == 3
        return prepareImage!(dest_arr, colorview(RGB, PermutedDimsArray(img, [3,1,2])), kern)
    else
        error("Array needs to match dimensions exactly, or 3rd dim should be of length 1 or 3 to allow colortype transformations")
    end
end

function prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:ImageCore.Colorant}
    # @show typeof(img), size(img)
    imgPerm = PermutedDimsArray(img, [2,1])  # Convert from column-major (julia default) to row-major (darknet)

    img_resized_size = sizethatfits(size(imgPerm), size(dest_arr))
    padding = getpadding(img_resized_size, size(dest_arr))
    xidx = (padding[1]+1):(size(dest_arr, 1) - padding[3])
    yidx = (padding[2]+1):(size(dest_arr, 2) - padding[4])
    target_img_subregion = view(dest_arr, xidx, yidx, :)
    if any(size(imgPerm)[1:2] .> size(dest_arr)[1:2]) #Apply blur first if reducing size, to avoid aliasing
        imgblur = ImageFiltering.imfilter(imgPerm, kern, NA())
        imgr = ImageTransformations.imresize(imgblur, img_resized_size[1:2])
        img_chv = channelview(imgr)
        if ndims(img_chv) == 2
            img_ready = repeat(Float32.(reshape(img_chv,(size(img_chv)...,1))),
                        outer=[1,1,size(dest_arr, 3)])
        elseif (size(img_chv,1) != size(dest_arr, 3)) && (size(img_chv, 1) == 1)
            img_ready = repeat(Float32.(PermutedDimsArray(img_chv, [2,3,1])),
                        outer=[1,1,size(dest_arr, 3)])
        else
            img_ready = PermutedDimsArray(img_chv, [2,3,1])[:, :, 1:size(dest_arr,3)]
        end
    elseif any(size(imgPerm)[1:2] .< size(dest_arr)[1:2])
        imgr = ImageTransformations.imresize(imgPerm, img_resized_size[1:2])
        img_chv = channelview(imgr)
        if ndims(img_chv) == 2
            img_ready = Float32.(reshape(img_chv, (size(img_chv)...,1)))
        else
            img_ready = view(PermutedDimsArray(Float32.(img_chv), [2,3,1]), :, :, 1:size(dest_arr,3))
        end
    else
        img_chv = channelview(imgPerm)
        if ndims(img_chv) == 3
            img_ready = view(PermutedDimsArray(Float32.(img_chv), [2,3,1]), :, :, 1:size(dest_arr,3))
        else
            img_ready = Float32.(repeat(PermutedDimsArray(img_chv, [1,2]), outer = [1, 1, size(dest_arr,3)]))
        end
    end
    target_img_subregion .= gpu(img_ready)
    scaled_padding = padding ./ size(dest_arr)[[1,2,1,2]]
    return (dest_arr, scaled_padding)
end
