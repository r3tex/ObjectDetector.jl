module BenchmarkExt

using ObjectDetector, BenchmarkTools, PrettyTables
using ObjectDetector: YOLO, emptybatch, prepare_image
using ObjectDetector.Colors: RGB

function ObjectDetector.benchmark(; models = sort(collect(keys(YOLO.YOLO_MODELS))), reverseAfter::Bool = false, img = rand(RGB,416,416), verbose=true, kw...)
    reverseAfter && (models = vcat(models, reverse(models)))

    header = ["Model", "loaded?", "load time (s)", "#results", "run time (s)", "run time (fps)", "allocations"]
    table = Array{Any}(undef, length(models), 7)
    for (i, modelname) in pairs(models)
        verbose && @info "Loading and running $modelname"
        table[i,:] = [modelname false "-" "-" "-" "-" "-"]

        loaded = true
        t_load = @elapsed begin
            mod = try
                YOLO.yolo_model(modelname; silent=true, kw...)
            catch ex
                loaded = false
                @warn "Failed to load $modelname: $ex"
            end
        end
        table[i, 2] = loaded
        loaded || continue

        table[i, 3] = round(t_load, digits=3)

        batch = emptybatch(mod)
        batch[:,:,:,1], padding = prepare_image(img, mod)

        res = mod(batch; detect_thresh=0.0, overlap_thresh=1.0) #run once
        t_run = @belapsed $mod($batch; detect_thresh=0.0, overlap_thresh=1.0);
        t_allocs = @allocated mod(batch; detect_thresh=0.0, overlap_thresh=1.0)
        table[i, 4] = size(res, 2)
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = round(1/t_run, digits=1)
        table[i, 7] = Base.format_bytes(t_allocs)

        mod = nothing
        batch = nothing
        GC.gc()
    end
    if pkgversion(PrettyTables) >= v"3"
        pretty_table(table; column_labels = header)
    else
        pretty_table(table; header = header)
    end
end

end # module
