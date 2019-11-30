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
function sizethatfits(src, dst)
    scale = minimum([dst[1]/src[1], dst[2]/src[2]])
    result = floor.(Int, src[[1,2]] .* scale)
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
    padding = round.(Int, [size_diff[1]/2, size_diff[2]/2, size_diff[1]/2, size_diff[2]/2])
end

"""
    prepareImage(img::AbstractArray{T}, model::U) where {T<:ImageCore.Colorant, U<:Model}
    prepareImage(img::AbstractArray{T}, img_resized_size::Tuple, kern) where {T<:ImageCore.Colorant}
    prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:Real}
    prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:ImageCore.Colorant}

Loads and prepares (resizes + pads) an image to fit within a given shape.
Input images should be column-major (julia default), and will be converted to row-major (darknet).
"""
function prepareImage(img::AbstractArray{T}, model::U) where {T<:ImageCore.Colorant, U<:Model}
    modelInputSize = getModelInputSize(model)
    if size(img) == modelInputSize[1:2]
        #TODO: Generalize for number of channels (i.e. some models trained on 1 channel)
        return permutedims(Float32.(channelview(img)), [3,2,1])[:,:,1:3], [0,0,0,0]
    end
    img_size = size(img)[[2,1]]
    img_resized_size = sizethatfits(img_size, modelInputSize)
    kern = resizekern(img_size, img_resized_size)
    return prepareImage(img, modelInputSize, kern)
end

prepareImage(img::AbstractArray{T}, modelInputSize::Tuple, kern) where {T<:ImageCore.Colorant} =
    prepareImage!(zeros(Float32, modelInputSize[1:3]), img, kern)

function prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:Real}
    size(img)[2,1,3] == size(dest_arr) && return permuteddimsview(img, [2,1,3]), [0,0,0,0]

    size(img,3) == 1 && return prepareImage!(dest_arr, colorview(Gray, img), kern)

    size(img,3) == 3 && return prepareImage!(dest_arr, colorview(RGB, permuteddimsview(img, [3,1,2])), kern)

    error("Array needs to match dimensions exactly, or 3rd dim should be of length 1 or 3 to allow colortype transformations")
end

function prepareImage!(dest_arr::AbstractArray{Float32}, img::AbstractArray{T}, kern) where {T<:ImageCore.Colorant}

    imgPerm = permuteddimsview(img, [2,1])  # Convert from column-major (julia default) to row-major (darknet)

    img_resized_size = sizethatfits(size(imgPerm), size(dest_arr))
    padding = getpadding(img_resized_size, size(dest_arr))
    xidx = (padding[1]+1):(size(dest_arr, 1) - padding[3])
    yidx = (padding[2]+1):(size(dest_arr, 2) - padding[4])
    target_img_subregion = view(dest_arr, xidx, yidx, :)
    if any(size(imgPerm)[1:2] .> size(dest_arr)[1:2]) #Apply blur first if reducing size, to avoid aliasing
        imgblur = ImageFiltering.imfilter(imgPerm, kern, NA())
        imgr = ImageTransformations.imresize(imgblur, img_resized_size[1:2])
        img_chv = Float32.(permuteddimsview(channelview(imgr), [2,3,1]))
        if (size(img_chv,3) != size(dest_arr, 3)) && (size(img_chv, 3) == 1)
            img_ready = repeat(img_chv, outer=[1,1,size(dest_arr, 3)])
        else
            img_ready = view(img_chv, :, :, 1:size(dest_arr,3))
        end
    elseif any(size(imgPerm)[1:2] .< size(dest_arr)[1:2])
        imgr = ImageTransformations.imresize(imgPerm, img_resized_size[1:2])
        img_ready = view(permuteddimsview(Float32.(channelview(imgr)), [2,3,1]), :, :, 1:size(dest_arr,3))
    else
        img_ready = view(permuteddimsview(Float32.(channelview(imgPerm)), [2,3,1]), :, :, 1:size(dest_arr,3))
    end
    target_img_subregion .= gpu(img_ready)
    scaled_padding = padding ./ size(dest_arr)[[1,2,1,2]]
    return dest_arr, scaled_padding
end
