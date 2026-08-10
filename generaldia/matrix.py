import torch

def unpack_real_symmetric(packed,n):
    packed=torch.as_tensor(packed)
    expected=n*(n+1)//2
    if packed.shape[-1]!=expected: raise ValueError(f"expected {expected} parameters")
    H=packed.new_zeros(packed.shape[:-1]+(n,n))
    k=0
    for i in range(n):
        for j in range(i,n):
            H[...,i,j]=packed[...,k]; H[...,j,i]=packed[...,k]; k+=1
    return H

def complex_hermitian_from_parts(diagonal, real_upper, imag_upper):
    diagonal=torch.as_tensor(diagonal)
    n=diagonal.shape[-1]
    expected=n*(n-1)//2
    if real_upper.shape[-1]!=expected or imag_upper.shape[-1]!=expected:
        raise ValueError("wrong number of off-diagonal parameters")
    dtype=torch.complex128 if diagonal.dtype==torch.float64 else torch.complex64
    H=torch.zeros(diagonal.shape[:-1]+(n,n),dtype=dtype,device=diagonal.device)
    H.diagonal(dim1=-2,dim2=-1).copy_(diagonal.to(dtype))
    k=0
    for i in range(n):
        for j in range(i+1,n):
            z=real_upper[...,k].to(dtype)+1j*imag_upper[...,k].to(dtype)
            H[...,i,j]=z; H[...,j,i]=z.conj(); k+=1
    return H
