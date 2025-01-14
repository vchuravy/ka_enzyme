using Test
using Enzyme
using KernelAbstractions
using CUDA

@kernel function square!(A)
    I = @index(Global, Linear)
    @inbounds A[I] *= A[I]
end

function square_caller(A, backend)
    kernel = square!(backend)
    kernel(A, ndrange=size(A))
end


@kernel function mul!(A, B)
    I = @index(Global, Linear)
    @inbounds A[I] *= B
end

function mul_caller(A, B, backend)
    kernel = mul!(backend)
    kernel(A, B, ndrange=size(A))
end

function enzyme_testsuite(backend, ArrayT, supports_reverse=true)
    @testset "kernels" begin
        A = ArrayT{Float64}(undef, 64)
        dA = ArrayT{Float64}(undef, 64)

        if supports_reverse

            A .= (1:1:64)
            dA .= 1

            Enzyme.autodiff(Reverse, square_caller, Duplicated(A, dA), Const(backend()))
            KernelAbstractions.synchronize(backend())
            @test all(dA .≈ (2:2:128))


            A .= (1:1:64)
            dA .= 1

            _, dB, _ = Enzyme.autodiff(Reverse, mul_caller, Duplicated(A, dA), Active(1.2), Const(backend()))[1]
            KernelAbstractions.synchronize(backend())

            @test all(dA .≈ 1.2)
            @test dB ≈ sum(1:1:64)
        end

        A .= (1:1:64)
        dA .= 1

        Enzyme.autodiff(Forward, square_caller, Duplicated(A, dA), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @test all(dA .≈ 2:2:128)

    end
end

@assert CUDA.functional()
@assert CUDA.has_cuda_gpu()
# enzyme_testsuite(CPU, Array, true)
# enzyme_testsuite(CUDABackend, CuArray, false)
enzyme_testsuite(CUDABackend, CuArray, true)