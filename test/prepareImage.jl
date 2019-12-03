@testset "prepareImage() input varieties" begin
    yolomod = YOLO.v3_tiny_COCO(w=416, h=416, silent=true)
    batch = emptybatch(yolomod)
    @testset "2D float32" begin
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 416, 416), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 500, 500), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 200, 500), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 500, 200), yolomod)
        @test true
    end
    @testset "3D float32 with 1 channel" begin
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 416, 416, 1), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 500, 500, 1), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 200, 500, 1), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 500, 200, 1), yolomod)
        @test true
    end
    @testset "3D float32 with 3 channels" begin
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 416, 416, 3), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 500, 500, 3), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 200, 500, 3), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Float32, 500, 200, 3), yolomod)
        @test true
    end
    @testset "2D Gray" begin
        batch[:,:,:,1], padding = prepareImage(rand(Gray, 416, 416), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Gray, 500, 500), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Gray, 200, 500), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(Gray, 500, 200), yolomod)
        @test true
    end
    @testset "2D RGB" begin
        batch[:,:,:,1], padding = prepareImage(rand(RGB, 416, 416), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(RGB, 500, 500), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(RGB, 200, 500), yolomod)
        batch[:,:,:,1], padding = prepareImage(rand(RGB, 500, 200), yolomod)
        @test true
    end
end
