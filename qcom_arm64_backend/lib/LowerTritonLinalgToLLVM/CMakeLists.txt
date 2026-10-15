add_mlir_conversion_library(LowerTritonLinalgToLLVM
        ARM64MathApproximation.cpp
        ARM64ProfInstrumentation.cpp
        LowerTmTensor.cpp
	LowerTritonLinalgToLLVM.cpp
        LowerConstantsSeparately.cpp

        LINK_COMPONENTS
        Core

        DEPENDS
	LowerTritonLinalgToLLVMPassIncGen

        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRDialectUtils
        MLIRIR
        MLIRMathDialect
        MLIRMathTransforms
        MLIRPass
        MLIRSupport
        MLIRTensorDialect
        MLIRTransforms
        MLIRTargetLLVMIRExport
        TritonLLVMIR
        )
