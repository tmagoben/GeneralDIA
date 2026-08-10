import torch
from generaldia.observables import derivative_matrix_elements
torch.set_default_dtype(torch.float64)
class LinearModel(torch.nn.Module):
 def forward(self,Z,R):
  x=R[0,0]; return torch.stack((torch.stack((0.05*x,torch.tensor(0.01))),torch.stack((torch.tensor(0.01),-0.05*x))))
def test_derivative_matrix_element_magnitude():
 Z=torch.tensor([1]); R=torch.tensor([[0.3,0.,0.]]) ; E,U,N=derivative_matrix_elements(LinearModel(),Z,R); expected=torch.tensor(0.05*0.01/(((0.05*0.3)**2+0.01**2)**0.5)); assert torch.allclose(torch.abs(N[0,0,0,1]),expected,atol=1e-10)
