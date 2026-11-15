"""Differentiable local point-to-triangle penetration for oriented closed meshes.
Nearest triangle candidates are discrete; distances differentiate both surfaces.
This is a local signed-distance approximation, independently checked with segment
intersection tests after fitting. It is not a collision-free guarantee.
"""
import torch
from pytorch3d.ops import knn_points

def signed_mesh_distance(points, vertices, faces, candidates=12):
    triangles=vertices[:,faces]
    with torch.no_grad():
        indices=knn_points(points.detach(),triangles.detach().mean(2),K=min(candidates,len(faces))).idx
    batch=torch.arange(len(points),device=points.device)[:,None,None]
    tri=triangles[batch,indices];p=points[:,:,None]
    a,b,c=tri.unbind(-2);ab=b-a;ac=c-a;n=torch.cross(ab,ac,dim=-1);nn=n.square().sum(-1).clamp(min=1e-16)
    plane=p-(((p-a)*n).sum(-1)/nn)[...,None]*n
    ap=plane-a;d00=ab.square().sum(-1);d01=(ab*ac).sum(-1);d11=ac.square().sum(-1);d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1);den=(d00*d11-d01.square()).clamp(min=1e-16)
    u=(d11*d20-d01*d21)/den;v=(d00*d21-d01*d20)/den;inside=(u>=0)&(v>=0)&(u+v<=1)
    closest=[]
    for x,y in [(a,b),(b,c),(c,a)]:
        e=y-x;t=(((p-x)*e).sum(-1)/e.square().sum(-1).clamp(min=1e-16)).clamp(0,1);closest.append(x+t[...,None]*e)
    edges=torch.stack(closest,-2);edge_index=(edges-p[...,None,:]).square().sum(-1).argmin(-1)
    edge=edges.gather(-2,edge_index[...,None,None].expand(*edge_index.shape,1,3)).squeeze(-2)
    q=torch.where(inside[...,None],plane,edge);squared=(p-q).square().sum(-1);best=squared.argmin(-1)
    distance=squared.gather(-1,best[...,None]).squeeze(-1).clamp(min=1e-16).sqrt()
    sign=((p-q)*n).sum(-1).gather(-1,best[...,None]).squeeze(-1).detach().sign()
    # A point outside a closed mesh bounding box is necessarily outside.
    # Nearest-face normals alone are ambiguous far from a concave surface.
    exterior=((points<vertices.amin(1)[:,None])|(points>vertices.amax(1)[:,None])).any(-1)
    return torch.where(exterior,distance,distance*sign)
