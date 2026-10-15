add_mlir_conversion_library(OptimizeTritonShared
        ARM64MatmulToMmt4DPass.cpp
        ARM64TilingPass.cpp
        ARM64VectorizationPass.cpp
        ARM64VectorLoweringPass.cpp
        BubbleUpExtractSlicePass.cpp
        OptimizeTritonShared.cpp
        PuntBufferPass.cpp
        RVOPass.cpp

        LINK_COMPONENTS
        Core

        DEPENDS
	OptimizeTritonSharedPassIncGen

        LINK_LIBS PUBLIC
        MLIRBufferizationDialect
        MLIRBufferizationPipelines
        MLIRBufferizationTransforms
        MLIRIR
        MLIRMemRefDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRPass
        MLIRSCFTransforms
        MLIRSupport
        MLIRTensorTransforms
        MLIRTransforms
        MLIRTargetLLVMIRExport
        MLIRVectorTransforms
        )
