@testset "Drawing boxes" begin
    # Testing that boxes are drawn as expected

    ### square model
    yolomod = YOLO.v3_tiny_COCO(w=416, h=416, silent=true)
    #Nonsquare low aspect ratio image
    img = fill(Gray(1), 200, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #Nonsquare high aspect ratio image
    img = fill(Gray(1), 100, 200)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #Square image, square model
    img = fill(Gray(1), 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    ### nonsquare low aspect ratio model
    yolomod = YOLO.v3_tiny_COCO(w=512, h=416, silent=true)
    #Square image
    img = fill(Gray(1), 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #nonsquare low aspect ratio image
    img = fill(Gray(1), 200, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    ### nonsquare high aspect ratio model
    yolomod = YOLO.v3_tiny_COCO(w=416, h=512, silent=true)
    #Square image
    img = fill(Gray(1), 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    #nonsquare low aspect ratio image
    img = fill(Gray(1), 100, 200)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepareImage(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = drawBoxes(img, yolomod, padding, res)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

    ## Non-transposed
    imgboxes = drawBoxes(collect(img'), yolomod, padding, res, transpose=false)
    @test all(imgboxes[1,:] .== Gray(0))
    @test all(imgboxes[:,1] .== Gray(0))
    @test all(imgboxes[end,:] .== Gray(0))
    @test all(imgboxes[:,end] .== Gray(0))

end
