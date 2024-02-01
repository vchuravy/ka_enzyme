using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore.EnzymeRules
using CUDA
using KernelAbstractions

@kernel function square!(A)
    I = @index(Global, Linear)
    @inbounds A[I] *= A[I]
end

target = CUDA.GPUCompiler.PTXCompilerTarget(v"7.5.0", v"7.5.0", true, nothing, nothing, nothing, nothing, false, nothing, nothing)
params = CUDA.CUDACompilerParams(v"7.5.0", v"8.2.0")
config = CUDA.CompilerConfig(target, params; kernel=true, name=nothing, always_inline=false)

mi = CUDA.methodinstance(typeof(()->return), Tuple{})
job = CUDA.CompilerJob(mi, config)

ModifiedBetween = Val{(false, false, false)}()
FT = Const{typeof(gpu_square!)}
ctxTy = KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize, KernelAbstractions.NDIteration.DynamicCheck, Nothing, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}, KernelAbstractions.NDIteration.NDRange{1, KernelAbstractions.NDIteration.DynamicSize, KernelAbstractions.NDIteration.DynamicSize, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}}}
args2T = (Duplicated{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}},)


TapeType = EnzymeCore.tape_type(job, ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), FT, Const,  Const{ctxTy}, args2T...)
@show TapeType