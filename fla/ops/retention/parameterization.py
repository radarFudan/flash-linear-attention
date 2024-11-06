import triton
import triton.language as tl

@triton.jit
def parameterization_in(lambdas):
    return tl.math.sqrt(lambdas / (1-lambdas))

@triton.jit
def parameterization_out(weights):
    return 1 - 1 / (1 + weights)


if __name__ == "__main__":
    lambdas = 0.999
    print(parameterization_in(lambdas))
    print(parameterization_out(parameterization_in(lambdas)))