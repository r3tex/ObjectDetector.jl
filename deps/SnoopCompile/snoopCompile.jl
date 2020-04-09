using SnoopCompile

@snoopi_bot "ObjectDetector" begin
    using ObjectDetector
    yolomod = YOLO.YOLO.v3_320_COCO()
    # batch = emptybatch(yolomod)
    # yolomod(batch)
end
