@testset "Drawing boxes" begin
    # Testing that boxes are drawn as expected

    ### square model
    yolomod = YOLO.v3_tiny_COCO(w=416, h=416, silent=true)
    #Nonsquare low aspect ratio image
    img = fill(Gray(1), 200, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    #Nonsquare high aspect ratio image
    img = fill(Gray(1), 100, 200)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    #Square image, square model
    img = fill(Gray(1), 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    ### nonsquare low aspect ratio model
    yolomod = YOLO.v3_tiny_COCO(w=512, h=416, silent=true)
    #Square image
    img = fill(Gray(1), 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    #nonsquare low aspect ratio image
    img = fill(Gray(1), 200, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    ### nonsquare high aspect ratio model
    yolomod = YOLO.v3_tiny_COCO(w=416, h=512, silent=true)
    #Square image
    img = fill(Gray(1), 100, 100)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    #nonsquare low aspect ratio image
    img = fill(Gray(1), 100, 200)
    batch = emptybatch(yolomod)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)
    res = collect([ padding[1] padding[2] 1.0-padding[3] 1.0-padding[4] 0.0 0.0;]') #note the transpose!
    imgboxes = draw_boxes(img, yolomod, padding, res, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

    ## Non-transposed
    imgboxes = draw_boxes(collect(img'), yolomod, padding, res, transpose=false, opacity=1.0)
    @test all(imgboxes[1,:] .== imgboxes[1,1])
    @test all(imgboxes[:,1] .== imgboxes[1,1])
    @test all(imgboxes[end,:] .== imgboxes[end,end])
    @test all(imgboxes[:,end] .== imgboxes[end,end])

end
