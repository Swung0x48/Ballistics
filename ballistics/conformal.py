"""Conformal road geometry: triangle unions and analytic footprint clipping.

The collision mesh remains authoritative.  Triangles triangulate the polygon
(including holes); internal diagonals are not support events.
"""
from collections import defaultdict, deque
from itertools import combinations
from dataclasses import dataclass, field
import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import nearest_points
from shapely.affinity import translate
import trimesh

from .coarse import WALK_NY
from .physics import WOOD

XZ = [0, 2]
# MEASURED: the exported L3 ramp has up to 0.011 height residual and 0.26
# degrees between adjacent strips. Compare with the 0.0099 contact skin.
ANGLE = math.cos(math.radians(.3))
PLANE_ERROR = .02
# Shared vertices in the collision export (same weld as floor adjacency).
WELD = 5
EPS = 1.e-8


def edge_key(a, b):
    return tuple(sorted((tuple(np.round(a, WELD)), tuple(np.round(b, WELD)))))


@dataclass
class Patch:
    normal: np.ndarray
    offset: float                # normal @ point = offset (upward normal)
    faces: tuple
    boundary: tuple              # (original face, a, b), oriented out of union
    polygon: object              # Shapely Polygon/MultiPolygon in world XZ
    low: np.ndarray
    high: np.ndarray

    def heights(self,radius=0.):
        a,b=self.low[XZ],self.high[XZ]
        xy=np.where(self.normal[XZ]>0,b,a)
        rise=radius*self.normal[1]
        return float(self.height(xy)+rise),float(self.height(a+b-xy)+rise)

    def height(self, xz, radius=0.):
        return (self.offset+radius-np.asarray(xz)@self.normal[XZ])/self.normal[1]


@dataclass
class Surface:
    patches: tuple
    face_patch: np.ndarray
    node_patch: np.ndarray
    triangles: np.ndarray
    rails: tuple
    regions: dict = field(default_factory=dict)
    node_patches: dict = field(default_factory=dict)


def build(mesh, rails=()):
    triangles = np.stack((mesh.A, mesh.B, mesh.C), axis=1)
    normal = np.asarray(mesh.N)
    eligible = np.flatnonzero((normal[:,1] >= WALK_NY) & ~mesh.rail & ~mesh.placed)
    face_patch = np.full(len(triangles), -1, dtype=int)
    if not len(eligible):
        return Surface((),face_patch,np.empty(0,dtype=int),triangles,tuple(rails))
    tm = trimesh.Trimesh(vertices=triangles[eligible].reshape(-1,3),
                         faces=np.arange(3*len(eligible)).reshape(-1,3),process=False)
    tm.merge_vertices(digits_vertex=WELD,merge_norm=True,merge_tex=True)
    # graph.facets' radius/span threshold is neither an angle nor a distance,
    # and has no common-plane check. Use its underlying adjacency/components
    # and the same boundary-edge primitive, with our own explicit predicate.
    pairs=tm.face_adjacency
    n0,n1=normal[eligible[pairs[:,0]]],normal[eligible[pairs[:,1]]]
    pairs=pairs[np.einsum('ij,ij->i',n0,n1)>=ANGLE]
    components=trimesh.graph.connected_components(pairs,nodes=np.arange(len(eligible)),min_len=1)
    graph=defaultdict(set)
    for i,j in pairs:
        graph[i].add(j); graph[j].add(i)
    patches=[]
    triangles_2d=tm.triangles[:,:,XZ]
    for component in sorted(components,key=lambda x: int(x.min())):
        for seed in sorted(component):
            source=eligible[seed]
            if face_patch[source]>=0:continue
            n=normal[source]; origin=triangles[source,0]
            todo=deque([seed]); checked={seed}; rows=[]
            while todo:
                row=todo.popleft(); i=eligible[row]
                if face_patch[i]>=0 or normal[i]@n<ANGLE or np.max(abs((triangles[i]-origin)@n))>PLANE_ERROR:
                    continue
                face_patch[i]=len(patches); rows.append(row)
                for j in sorted(graph[row]-checked):
                    checked.add(j); todo.append(j)
            faces=eligible[rows]
            # Same edge counting as Trimesh.facets_boundary. Keep the source
            # face and winding so rolling can resolve a real neighbouring facet.
            es=tm.edges_sorted.reshape(-1,3,2)[rows].reshape(-1,2)
            boundary=[]
            for j in trimesh.grouping.group_rows(es,require_count=1):
                row,k=divmod(int(j),3); i=faces[row]
                a,b=triangles[i,k],triangles[i,(k+1)%3]
                if np.cross(b-a,triangles[i,(k+2)%3]-a)@n<0.:a,b=b,a
                boundary.append((i,a,b))
            polygon=shapely.union_all([Polygon(t) for t in triangles_2d[rows]])
            points=triangles[faces].reshape(-1,3)
            patches.append(Patch(n,float(n@origin),tuple(faces),tuple(boundary),polygon,points.min(0),points.max(0)))
    return Surface(tuple(patches),face_patch,np.empty(0,dtype=int),triangles,tuple(rails))


def clip(polygon, normals, bounds):
    """Clip a convex polygon by linear half-planes without sampling."""
    p=np.asarray(polygon)
    for n,h in zip(normals,bounds):
        if not len(p): break
        result=[]; signed=p@n-h
        for i,(a,b) in enumerate(zip(p,np.roll(p,-1,axis=0))):
            x,y=signed[i],signed[(i+1)%len(signed)]
            if (x<=EPS)!=(y<=EPS):
                # Clip the same numerically enlarged half-space used for signs.
                result.append(a+((x-EPS)/(x-y))*(b-a))
            if y<=EPS: result.append(b)
        p=np.asarray(result).reshape(-1,2)
    return p


def drift_rows(patch, radius, origin, direction, slope, ratio, cap):
    h0=origin[1]-(patch.offset+radius)/patch.normal[1]
    dh=patch.normal[XZ]/patch.normal[1]
    rows=[(-1.,np.zeros(2),0.),(1.,np.zeros(2),cap),(-slope,-dh,h0)]
    for i,basis in enumerate(np.eye(2)):
        for sign in (-1.,1.):
            rows.append((-sign*direction[i]-ratio*slope,sign*basis-ratio*dh,sign*origin[XZ][i]+ratio*h0))
    return rows


def drift_planes(patch, radius, origin, direction, slope, ratio, cap):
    """Project the linear reach constraints along the drift variable."""
    rows=drift_rows(patch,radius,origin,direction,slope,ratio,cap)
    lower=[r for r in rows if r[0]<-EPS]; upper=[r for r in rows if r[0]>EPS]
    planes=[(a,b) for c,a,b in rows if abs(c)<=EPS]
    planes.extend((a/u-d/l,b/u-e/l) for u,a,b in upper for l,d,e in lower)
    return tuple(np.array(i) for i in zip(*planes))


def speed_witnesses(patch, pieces, radius, origin, direction, slope, ratio, cap, vmax, loss, dt, gravity, coeff):
    """Solve the curved speed cap on the projected (drift, drop) polytope.

    The maximum-speed height bound G*B(s/vmax)-slope*s is convex.
    A linear edge of the upper hull has constant slope; the maximum slack
    occurs at an endpoint or the single PSI kink where its derivative
    overtakes that slope. Find the kink algebraically, not by iteration.
    """
    if vmax<=EPS:return
    rows=drift_rows(patch,radius,origin,direction,slope,ratio,cap)
    base=np.array([(*a,c) for c,a,_ in rows]); limits=np.array([b for _,_,b in rows]); logd=math.log1p(-loss)
    for piece in pieces:
        for part in polygon_parts(piece):
            if part.geom_type!='Polygon':continue
            for tile in shapely.constrained_delaunay_triangles(part).geoms:
                tri=np.asarray(shapely.orient_polygons(tile).exterior.coords)[:-1]
                diff=np.roll(tri,-1,axis=0)-tri
                n=np.c_[diff[:,1],-diff[:,0]]
                m=np.vstack((base,np.c_[n,np.zeros(len(n))])); h=np.r_[limits,np.sum(n*tri,axis=1)]
                triples=np.asarray(list(combinations(range(len(m)),3)))
                mats=m[triples]; good=abs(np.linalg.det(mats))>1.e-12
                vertices=np.linalg.solve(mats[good],h[triples[good],None]).squeeze(-1)
                vertices=vertices[np.all(vertices@m.T<=h+1.e-7,axis=1)]
                if not len(vertices):continue
                drop=origin[1]-patch.height(vertices[:,:2],radius)
                # Project the polytope; only its upper boundary can minimize
                # the convex height constraint. Keep the original 3D witness.
                first={}
                for v,y in zip(vertices,drop):first.setdefault((v[2],y),v)
                hull=MultiPoint(list(first)).convex_hull
                boundary=(list(hull.exterior.coords) if hull.geom_type=='Polygon' else list(hull.coords))
                targets=[first[s] for s in boundary]
                for a,b in zip(boundary,boundary[1:]):
                    if abs(b[0]-a[0])<=EPS:continue
                    secant=(b[1]-a[1])/(b[0]-a[0])
                    x=loss*vmax*(secant+slope)/gravity
                    if x<=0.:continue
                    k=math.ceil(math.log1p(x)/-logd)
                    s=vmax*dt*(-math.expm1(logd*k))/loss
                    f=(s-a[0])/(b[0]-a[0])
                    if 0.<f<1.:targets.append((1-f)*first[a]+f*first[b])
                for x,z,s in targets:
                    if s<0. or loss*s>=vmax*dt:continue
                    k=math.floor(math.log1p(-loss*s/(vmax*dt))/logd)
                    A,B=coeff(k,loss); A1,B1=coeff(k+1,loss)
                    if A1<=A:continue
                    Bg=B+(B1-B)*(s/vmax-A)/(A1-A)
                    y=patch.height([x,z],radius)
                    if origin[1]-y+slope*s+EPS>=gravity*Bg:yield np.array([x,y,z])


def polygon_parts(g):
    if g.is_empty:return
    if hasattr(g,'geoms'):
        for p in g.geoms:yield from polygon_parts(p)
    else:
        yield g


def footprint(polygon, cuts):
    """Intersect the true polygon with a bounded intersection of half-planes."""
    x,z,X,Z=polygon.bounds
    pad=1.e-6
    frame=np.array([[x-pad,z-pad],[X+pad,z-pad],[X+pad,Z+pad],[x-pad,Z+pad]])
    for n,h in cuts:
        p=clip(frame,n,h)
        if len(p):yield polygon.intersection(MultiPoint(p).convex_hull)


class Targets:
    """Block x rolling-region polygons; graph nodes only map witnesses."""
    def __init__(self, surface, pos, by_face, labels_by_ball, rest_count):
        self.surface=surface
        self.pos=pos
        self.lowest=min((float(p.height(np.where(p.normal[XZ]>0,p.high[XZ],p.low[XZ])))
                         for p in surface.patches),default=None)
        self.groups={}
        memberships=defaultdict(set)
        for name,labels in labels_by_ball.items():
            groups=[]
            for k,p in enumerate(surface.patches):
                nodes=defaultdict(set)
                for f in p.faces:
                    for i in by_face.get(f,()):
                        if i<rest_count:
                            nodes[int(labels[i])].add(i); memberships[i].add(k)
                if not nodes or p.polygon.is_empty:continue
                if len(nodes)==1:
                    l,ids=next(iter(nodes.items()))
                    groups.append((k,l,np.array(sorted(ids)),p.polygon))
                    continue
                # ASSUMED: the coarse graph tells which side a landing can
                # join, not the exact divider inside a facet. Use the nearest
                # supported node in XZ; split the actual polygon by Voronoi
                # cells. Coincident nodes may belong to more than one region.
                sites=defaultdict(set)
                for l,ids in nodes.items():
                    for i in ids:sites[tuple(pos[i,XZ]-WOOD.radius*p.normal[XZ])].add(l)
                tiles=defaultdict(list)
                if len(sites)==1:
                    for l in nodes:tiles[l].append(p.polygon)
                else:
                    ordered=shapely.geos_version >= (3,12,0)
                    cells=shapely.voronoi_polygons(MultiPoint(list(sites)),extend_to=box(*p.polygon.bounds),ordered=ordered)
                    pairs=(zip(sites.values(),cells.geoms) if ordered else
                           ((ls,next(cell for cell in cells.geoms if cell.covers(Point(x)))) for x,ls in sites.items()))
                    for ls,cell in pairs:
                        tile=p.polygon.intersection(cell)
                        if not tile.is_empty:
                            for l in ls:tiles[l].append(tile)
                for l,ids in sorted(nodes.items()):
                    polygon=shapely.union_all(tiles[l])
                    if not polygon.is_empty:groups.append((k,l,np.array(sorted(ids)),polygon))
            self.groups[name]=groups
        surface.regions=self.groups
        surface.node_patches={i:tuple(sorted(ps)) for i,ps in memberships.items()}
        surface.node_patch=np.full(len(pos),-1,dtype=int)
        for i, ps in surface.node_patches.items():surface.node_patch[i]=ps[0]
        self.unmapped=np.flatnonzero(surface.node_patch[:rest_count]<0)
        self.geometry={name:np.array([r[-1].bounds for r in gs]).reshape(-1,4) for name,gs in self.groups.items()}
        self.slopes={name:np.array([surface.patches[r[0]].normal[XZ] for r in gs]).reshape(-1,2)
                     for name,gs in self.groups.items()}

    def representatives(self, name, radius, lo, hi, planes, acceptable, *, stats, prefix, anchor=None,
                        exact=None, alternatives=False, spine=None, node_targets=None):
        groups=self.groups[name]; extent=self.geometry[name]
        shift=radius*self.slopes[name]
        indices=np.flatnonzero(np.all((extent[:,:2]+shift<=hi+EPS)&(extent[:,2:]+shift>=lo-EPS),axis=1))
        for j in indices:
            k,label,ids,polygon=groups[j]
            patch=self.surface.patches[k]
            stats[prefix+'_patches']+=1
            cuts=planes(patch)
            if not cuts:continue
            if np.any(shift[j]):polygon=translate(polygon,*shift[j])
            pieces=[p for p in footprint(polygon,cuts) if not p.is_empty]
            if not pieces:continue
            stats[prefix+'_intersections']+=1
            seen=set(); found=False
            for piece in pieces:
                for part in polygon_parts(piece):
                    points=[part.representative_point()]
                    if anchor is not None:points.insert(0,nearest_points(Point(anchor),part)[1])
                    if part.covers(part.centroid):points.append(part.centroid)
                    if spine is not None:
                        origin,direction=spine
                        x,z,X,Z=part.bounds
                        v=(np.array([[x,z],[x,Z],[X,z],[X,Z]])-origin)@direction
                        ray=LineString([origin+(v.min()-EPS)*direction,origin+(v.max()+EPS)*direction])
                        for line in polygon_parts(part.intersection(ray)):
                            if line.geom_type=='LineString':
                                a,b=np.asarray(line.coords)[[0,-1]]
                                points.extend(map(Point,(a,(a+b)*.5,b)))
                            elif line.geom_type=='Point':points.append(line)
                    # Actual vertices of the intersection, not spaced samples.
                    vertices=(part.exterior.coords[:-1] if part.geom_type=='Polygon' else part.coords)
                    found_part=False
                    for pt in points+list(map(Point,vertices)):
                        x=tuple(pt.coords[0])
                        if x in seen:continue
                        if found_part and pt not in points:continue
                        seen.add(x)
                        q=np.array([x[0],patch.height(x,radius),x[1]])
                        if not acceptable(q):continue
                        v=int(ids[np.argmin(np.sum((self.pos[ids]-q)**2,axis=1))])
                        yield v,q,k,label
                        if node_targets is not None:
                            mapped=node_targets[v]
                            xz=tuple(mapped[XZ])
                            # One mapped node per geometric witness; no search
                            # over the lattice as an independent target set.
                            # Check on the same clipped geometry and re-solve.
                            if xz not in seen and part.buffer(EPS).covers(Point(xz)) and acceptable(mapped):
                                seen.add(xz)
                                yield v,mapped,k,label
                        found=found_part=True
                        if not alternatives:break
                    if found and not alternatives:break
                if found and not alternatives:break
            if not found and exact is not None:
                for q in exact(patch,pieces):
                    if acceptable(q):
                        v=int(ids[np.argmin(np.sum((self.pos[ids]-q)**2,axis=1))])
                        yield v,q,k,label
                        break


def write_candidates(graph, stem):
    """Publish the aimed road points and mapped rail departures, not just winners."""
    import csv
    road=graph.info.get('conformal_launch_candidates',())
    rail=graph.info.get('rail_departure_candidates',())
    with stem.with_name(stem.name+'-conformal-candidates.csv').open('w',encoding='ascii',newline='') as f:
        w=csv.writer(f); w.writerow(('site','key','ball','patch','rolling_region','representative_node','x','y','z'))
        w.writerows(road)
    with stem.with_name(stem.name+'-conformal-rail-candidates.csv').open('w',encoding='ascii',newline='') as f:
        fields=('site','source','first_rail','target','ball','dest_component','target_patch','mode','rail','target_node',
                'launch','depart','depart_velocity','landing','push','interval','ticks','seconds')
        w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rail)
