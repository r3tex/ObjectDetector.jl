@testset "prepare_image() input varieties" begin
    yolomod = YOLO.v3_tiny_COCO(w=416, h=416, silent=true)
    batch = emptybatch(yolomod)
    @testset "2D float32" begin
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 416, 416), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 500, 500), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 200, 500), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 500, 200), yolomod)
        @test true
    end
    @testset "3D float32 with 1 channel" begin
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 416, 416, 1), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 500, 500, 1), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 200, 500, 1), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 500, 200, 1), yolomod)
        @test true
    end
    @testset "3D float32 with 3 channels" begin
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 416, 416, 3), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 500, 500, 3), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 200, 500, 3), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Float32, 500, 200, 3), yolomod)
        @test true
    end
    @testset "2D Gray" begin
        batch[:,:,:,1], padding = prepare_image(rand(Gray, 416, 416), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Gray, 500, 500), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Gray, 200, 500), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(Gray, 500, 200), yolomod)
        @test true
    end
    @testset "direct prepare_image! with matching-size 2D Float32" begin
        dest = zeros(Float32, 416, 416, 3)
        arr, padding = prepare_image!(dest, rand(Float32, 416, 416), nothing; use_gpu=false)
        @test size(arr) == (416, 416, 3)
        @test padding == [0, 0, 0, 0]
    end
    @testset "2D RGB" begin
        batch[:,:,:,1], padding = prepare_image(rand(RGB, 416, 416), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(RGB, 500, 500), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(RGB, 200, 500), yolomod)
        batch[:,:,:,1], padding = prepare_image(rand(RGB, 500, 200), yolomod)
        @test true
    end
end
