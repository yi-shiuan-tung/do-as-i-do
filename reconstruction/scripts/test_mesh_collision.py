"""Geometric regression checks for bidirectional thin-wall collisions."""
import numpy as np,torch,trimesh
from mesh_collision import signed_mesh_distance
m=trimesh.creation.box(extents=[.02,.02,.02]);v=torch.tensor(m.vertices,dtype=torch.float32)[None].requires_grad_();f=torch.tensor(m.faces,dtype=torch.long)
p=torch.tensor([[[0.,0.,0.],[.03,0,0],[.0005,0,0],[.5,.5,.5]]],requires_grad=True)
d=signed_mesh_distance(p,v,f);assert torch.allclose(d[0,:3],torch.tensor([-.01,.02,-.0095]),atol=1e-6);assert d[0,3]>0
# A thin plate crosses the box even though every box vertex is outside plate material.
plate=trimesh.creation.box(extents=[.001,.04,.04]);pv=torch.tensor(plate.vertices,dtype=torch.float32)[None];pf=torch.tensor(plate.faces,dtype=torch.long)
oneway=signed_mesh_distance(v,pv,pf);assert (oneway>0).all();assert d[0,2]<0
loss=torch.relu(-d).square().sum();loss.backward();assert torch.isfinite(p.grad).all() and torch.isfinite(v.grad).all();assert p.grad[0,2,0]<0
# A shared rigid transform must not change signed distances.
R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);t=torch.tensor([.2,.3,.4]);moved=signed_mesh_distance(p.detach()@R.T+t,v.detach()@R.T+t,f);assert torch.allclose(moved,d.detach(),atol=1e-6)
print('PASS: inside/outside, exterior guard, missed one-way thin-wall collision, rigid invariance, finite separating gradients')
