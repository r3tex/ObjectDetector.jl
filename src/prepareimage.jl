using ImageFiltering
using ImageTransformations
using ImageCore

"""
    resizekern(source_size::Tuple{Int,Int}, dest_size::Tuple{Int,Int})

Create an image resize blur kernel, for use before reducing image size to avoid aliasing.
"""
function resizekern(source_size, dest_size)
    # Blur before resizing to prevent aliasing (kernel size dependent on both source and target image size)
    σ = 0.75 .* (source_size[1:2] ./ dest_size[1:2])
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
function sizethatfits(original_size, target_shape)
    channels = length(target_shape) == 2 ? 1 : target_shape[3]
    if original_size[1] > original_size[2]
        target_img_size = (
            target_shape[1],
            floor(Int, target_shape[2] * (original_size[2] / original_size[1])),
            channels
        )
    else
        target_img_size = (
        floor(Int, target_shape[1] * (original_size[1] / original_size[2])),
        target_shape[2],
        channels
        )
    end
    return target_img_size
end

"""
    resizePadImage(img::Array{T}, model::yolo) where {T<:ImageCore.Colorant}
    resizePadImage(img::Array{T}, target_img_size::Tuple{Int,Int}, kern) where {T<:.Color}
    resizePadImage(img::Array{T}, target_img::Array{U}, kern) where {T<:Color, U<:Float32}

Loads and prepares (resizes + pads) an image to fit within a given shape.
Returns the image and the padding.
"""
function resizePadImage(img::Array{T}, model::U) where {T<:ImageCore.Colorant, U<:Model}
    modelInputSize = getModelInputSize(model)
    if size(img) == modelInputSize[1:2]
        #TODO: Generalize for number of channels (i.e. some models trained on 1 channel)
        return permutedims(Float32.(channelview(img)), [3,2,1])[:,:,1:3]
    end
    img_size = size(img)
    target_img_size = sizethatfits(img_size, modelInputSize)
    kern = resizekern(img_size, target_img_size)
    return resizePadImage(img, modelInputSize, kern)
end
function resizePadImage(img::Array{T}, target_img_size::Tuple, kern) where {T<:ImageCore.Colorant}
    target_img = zeros(Float32, target_img_size[1:3])
    return resizePadImage(img, target_img, kern)
end
function resizePadImage(img::Array{T}, target_img::Array{U}, kern) where {T<:ImageCore.Colorant, U<:Float32}
    target_img_size = sizethatfits(size(img), size(target_img))
    # Determine top and left padding
    vpad_top = floor(Int, (size(target_img,1) - target_img_size[1]) / 2)
    hpad_left = floor(Int, (size(target_img,2) - target_img_size[2]) / 2)

    # Determine bottom and right padding accounting for rounding of top and left (to ensure accuate result image size if source has odd dimensions)
    vpad_bottom = size(target_img,1) - (vpad_top + target_img_size[1])
    hpad_right = size(target_img,2) - (hpad_left + target_img_size[2])

    padding = [hpad_left, vpad_top, hpad_right, vpad_bottom]

    vindex = (vpad_bottom+1):(size(target_img, 1) - vpad_top)
    hindex = (hpad_left+1):(size(target_img, 2) - hpad_right)

    if size(img,1) > size(target_img, 1) #Apply blur first if reducing size, to avoid aliasing
        imgr = ImageTransformations.imresize(ImageFiltering.imfilter(img, kern, NA()), target_img_size[1:2])
        target_img[vindex, hindex, :] .= permutedims(Float32.(channelview(imgr)), [3,2,1])[:,:,1:size(target_img,3)]
    elseif size(img,1) < size(target_img, 1)
        imgr = ImageTransformations.imresize(img, target_img_size[1:2])
        target_img[vindex, hindex, :] .= permutedims(Float32.(channelview(imgr)), [3,2,1])[:,:,1:size(target_img,3)]
    else
        target_img[vindex, hindex, :] .= permutedims(Float32.(channelview(img)), [3,2,1])[:,:,1:size(target_img,3)]
    end

    return target_img
end
