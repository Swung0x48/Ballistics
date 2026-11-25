"""Conformal search domains, independent of the drawing/mapping lattice.

This is the topology and finite entrance-catalogue layer. In particular,
mesh adjacency edges below are NOT checked takeoff/carry witnesses.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import conformal, motion, physics as ph, rails, supportcarry as sc, sweep, terrain
from .coarse import WALK_NY

EPS = 1.e-7  # numerical comparison, not a physical threshold
XZ = [0, 2]


@dataclass(frozen=True, order=True)
class Road:
    ball: str
    region: int
    mechanism: tuple = ()


@dataclass(frozen=True, order=True)
class Rail:
    ball: str
    chain: int
    line: str
    piece: int
    direction: int


@dataclass(frozen=True)
class Seam:
    patches: tuple
    endpoints: np.ndarray
    faces: tuple


@dataclass(frozen=True)
class Region:
    node: Road
    patches: tuple
    rest_patches: tuple
    guarded_patches: tuple


@dataclass(frozen=True)
class Portal:
    source: Road | Rail
    target: Road | Rail
    kind: str
    patch_pair: tuple | None
    segment: np.ndarray


@dataclass
class Layout:
    roads: dict
    patch_nodes: dict
    rail_segments: dict
    portals: tuple
    seams: tuple
    verdicts: dict
    boundary: dict

    def mapped(self, ball, patches):
        """Mapping is many-to-many at a seam; it is never a search seed."""
        return tuple(sorted({self.patch_nodes[ball][k] for k in patches if k in self.patch_nodes[ball]}))


def welded(surface):
    """READ: welded positive-length spans of the exported conformal mesh."""
    incident, _ = terrain.neighbours(surface)
    out=[]; matched=set()
    for rows in incident.values():
        for (i,a,b,f),(j,x,y,g) in combinations(rows,2):
            if i==j or np.linalg.norm((b-a)[XZ])<=EPS or (b-a)[XZ]@(y-x)[XZ]>=0.:
                continue
            out.append(Seam((i,j),np.stack((a,b)),(f,g)))
            matched.update(((i,conformal.edge_key(a,b)),(j,conformal.edge_key(x,y))))
    boundary = {k:tuple((f,a,b) for f,a,b in p.boundary if (k,conformal.edge_key(a,b)) not in matched)
                for k,p in enumerate(surface.patches)}
    return tuple(out), boundary


def _capsule_blocked(tf, obs, a, b, radius, excluded):
    """Segment/triangle Voronoi cells + quadratic distance, no ray probes."""
    fs=tf._box_triangles((np.minimum(a,b)-radius)[None],(np.maximum(a,b)+radius)[None],
                         obs.lo,obs.hi,obs.mesh._grid)
    for f in fs:
        if f in excluded:
            continue
        tri=obs.tris[f]; normal=obs.mesh.N[f]
        for feature in range(7):
            lo,hi,w,d=sweep.feature_span(tri,normal,feature,a,b-a)
            lo,hi=sweep.sphere_span(w,d,radius,lo,hi)
            if lo<=hi:
                return True
    return False


def _obstacle_box(a,b,r,parts):
    # DERIVED conservative test in the part's local orthonormal chart.
    for part in parts:
        c=np.asarray(part['placed']); axes=np.asarray(part['axes']); half=np.asarray(part['half'])
        if motion.line_box_distance(axes@(a-c),axes@(b-a),half,0.,1.) <= r:
            return True
    return False


def _meets_patch(p,part,r):
    # Broad screen for same-patch mechanism cuts; marked unresolved rather
    # than silently turning the entire patch into a static rest area.
    extent=np.abs(np.asarray(part['axes'])).T@np.asarray(part['half'])+r
    center=np.asarray(part['placed']); a,b=center-extent,center+extent
    return bool(np.all(a<=p.high+r) and np.all(b>=p.low-r))


def _rest(ball,p):
    n=p.normal
    return n[1]>=WALK_NY and terrain.slope_verdict(ball,np.linalg.norm(n[XZ])/n[1],n[XZ])>0


def _seam_verdict(tf,obs,surface,seam,ball,lips,parts,guarded):
    i,j=seam.patches; p,q=(surface.patches[k] for k in seam.patches); a,b=seam.endpoints
    if not (_rest(ball,p) and _rest(ball,q)):
        return 'static_slope'
    if i in guarded or j in guarded:
        return 'mechanism_patch'
    if (i,j) in lips or (j,i) in lips:
        return 'step_lip'
    r=ball.radius+sc.SKIN
    n,m=p.normal,q.normal
    # DERIVED: two tangency planes intersect at r/cos(theta/2) along
    # the bisector. ASSUMED: consume at most the existing contact skin
    # for replacing this micro-transition by a smooth region membership.
    cosine=np.sqrt(max(0.,(1+float(n@m))*.5))
    if max(abs(p.height(x[XZ])-q.height(x[XZ])) for x in (a,b)) > conformal.PLANE_ERROR:
        return 'export_plane_mismatch'
    if cosine<=EPS or r*(1/cosine-1)>sc.SKIN:
        return 'normal_turn'
    axis=sc.unit(b-a)
    for patch,sign in ((p,1.),(q,-1.)):
        outward=sign*sc.unit(np.cross(axis,patch.normal))
        # The best legal world keys for crossing this particular finite seam.
        direction=outward[[0,2]]
        key=np.sign(direction)
        if sc.law(ball,patch.normal,key).impulse@outward<=EPS:
            return 'no_crossing_thrust'
    # ASSUMED representative crossing at the middle of the finite span; this
    # is a portal representative, not an assertion about an entire seam.
    # Exclude the two *bearing* patches for this local geometric screen only.
    mid=(a+b)*.5
    foot=lambda z: np.array([mid[0],z.height(mid[XZ]),mid[2]])
    start,end=foot(p)+r*n,foot(q)+r*m
    if _obstacle_box(start,end,r,parts) or any(motion.box_hit(motion.line(start,end),lo,hi) is not None for lo,hi in obs.boxes):
        return 'mechanism_or_death'
    if _capsule_blocked(tf,obs,start,end,ball.radius-sc.RECAPTURE,set(p.faces)|set(q.faces)):
        return 'mesh_blocker'
    return 'smooth'


def _tracks(metadata,ball):
    for i,lane in enumerate(metadata):
        rho=lane['radius']; center=lane['segments']
        if len(center)<2:
            continue
        if lane.get('double') and lane.get('axes'):
            a,b=[rails.straight_pieces(np.asarray(q)) for q in lane['axes']]
            if min(len(a),len(b))<2:
                continue
            tangent=sc.unit(center[-1]-center[0]); side=sc.unit(np.cross(_rail_normal(tangent),tangent))
            if (a[0]-b[0])@side<0.:
                a,b=b,a
            yield i,'groove',rails.paired_pieces(a,b,ball.radius+sc.SKIN,rho)
            for line,x in (('left_crown',a),('right_crown',b)):
                normals=[_rail_normal(sc.unit(d)) for d in np.diff(x,axis=0)]
                # Explicit endpoints per straight piece: turning at a ring
                # stays a portal, not a bend in a resampled ball-centre line.
                yield i,line,np.stack((x[:-1],x[1:]),axis=1)+ (ball.radius+rho+sc.SKIN)*np.asarray(normals)[:,None,:]
        else:
            lift=ball.radius+sc.SKIN-ph.WOOD.radius
            yield i,'single' if not lane['double'] else 'groove',center+[0.,lift,0.]


def _rail_normal(t):
    up=np.array([0.,1.,0.]); return sc.unit(up-t*t[1])


def build(tf,obs,surface,*,parts=(),metadata=None,lips=None,mechanism=()):
    """Catalogue shared geometry and candidate portals once per layout."""
    seams,boundary=welded(surface)
    if lips is None:
        lips=terrain.step_lips(obs.mesh,surface,terrain.neighbours(surface)[0])
    step_portals=tuple(lip for group in lips.values() for lip in group)
    lips={(lip.lower,lip.upper) for lip in step_portals}
    if metadata is None:
        metadata=rails.lanes(obs.mesh.tubes)
    roads={}; patch_nodes={}; rail_segments={}; portals=[]; verdicts={}
    for ball in ph.BALLS.values():
        parent=list(range(len(surface.patches)))
        def find(k):
            while parent[k]!=k:
                parent[k]=parent[parent[k]]; k=parent[k]
            return k
        guarded={k for k,p in enumerate(surface.patches) if any(_meets_patch(p,x,ball.radius) for x in parts)}
        outcomes=[]
        for seam in seams:
            i,j=seam.patches; code=_seam_verdict(tf,obs,surface,seam,ball,lips,parts,guarded); outcomes.append(code)
            if code=='smooth':
                a,b=sorted((find(i),find(j))); parent[b]=a
        groups=defaultdict(list); index={}
        for k in range(len(parent)):
            node=Road(ball.name,find(k),mechanism); groups[node].append(k); index[k]=node
        for node,ks in groups.items():
            roads[node]=Region(node,tuple(ks),tuple(k for k in ks if k not in guarded and _rest(ball,surface.patches[k])),
                               tuple(k for k in ks if k in guarded))
        patch_nodes[ball.name]=index; verdicts[ball.name]=tuple(outcomes)
        for seam,code in zip(seams,outcomes):
            i,j=seam.patches
            if index[i]!=index[j]:
                for u,v in ((i,j),(j,i)):
                    portals.append(Portal(index[u],index[v],code,(u,v),seam.endpoints))
        for lip in step_portals:
            u,v=lip.lower,lip.upper
            portals.append(Portal(index[u],index[v],'step_lip',(u,v),np.stack((lip.a,lip.b))))
        for chain,line,pts in _tracks(metadata,ball):
            spans=np.stack((pts[:-1],pts[1:]),axis=1) if pts.ndim==2 else pts
            for sign in (-1,1):
                nodes=[Rail(ball.name,chain,line,k,sign) for k in range(len(spans))]
                for node,seg in zip(nodes,spans):
                    rail_segments[node]=seg if sign>0 else seg[::-1]
                if sign<0:
                    nodes.reverse()
                for u,v in zip(nodes,nodes[1:]):
                    portals.append(Portal(u,v,'rail_ring',None,np.stack((rail_segments[u][1],rail_segments[v][0]))))
    return Layout(roads,patch_nodes,rail_segments,tuple(portals),seams,verdicts,boundary)


@dataclass(frozen=True)
class Entry:
    """Exact finite catalogue identity, never a rounded measured state.

    A catalogue producer supplies the semantics of the atom, e.g. an entire
    labelled positional/velocity family. Merely sharing a location bounding
    box or a destination does not make two entries equal.
    """
    node: Road | Rail
    family: str


class Worklist:
    """Finite union propagation over a frozen catalogue of entrance families.

    Transitions must have been derived for the whole incoming atom. A mapped
    graph edge with a checked representative does *not* supply that contract.
    """
    def __init__(self, catalogue, transfers, domains=None):
        self.catalogue=frozenset(catalogue)
        self.transfers={a:frozenset(bs) for a,bs in transfers.items()}
        self.domains={} if domains is None else dict(domains)
        if any(a not in self.catalogue or not self.catalogue.issuperset(bs) for a,bs in self.transfers.items()):
            raise ValueError('transfer outside frozen entrance catalogue')

    def propagate(self, seeds):
        seen=defaultdict(set); atoms=defaultdict(list); arrivals=[]; covered=[]; q=deque()
        def enter(atom,via):
            if atom not in self.catalogue:
                raise ValueError('seed outside frozen entrance catalogue')
            if atom.family in seen[atom.node]:
                covered.append((atom,via,atom)); return
            incoming=self.domains.get(atom)
            if incoming is not None:
                for prior in atoms[atom.node]:
                    current=self.domains.get(prior)
                    # A smaller arrival may have a separately checked suffix.
                    # Preserve it unless the compiled transfer also subsumes it.
                    if (current is not None and current.covers(incoming) and
                        self.transfers.get(prior,frozenset()) >= self.transfers.get(atom,frozenset())):
                        covered.append((atom,via,prior)); return
            seen[atom.node].add(atom.family); atoms[atom.node].append(atom)
            arrivals.append((atom,via)); q.append(atom)
        for a in seeds:
            enter(a,None)
        while q:
            a=q.popleft()
            for b in sorted(self.transfers.get(a,()),key=lambda e:(repr(e.node),e.family)):
                enter(b,a)
        return dict(seen=dict(seen),expanded=arrivals,covered=covered)

@dataclass(frozen=True)
class Domain:
    """An affine joint image with a proof context and its exact restrictions.

    Same position/velocity projections do not certify equal images. These
    tests recognize the common-chart case without geometric widening.
    """
    context: tuple
    center: tuple
    image: tuple
    equal: frozenset
    upper: frozenset

    @classmethod
    def from_family(cls,family,context):
        def constraints(rows,rhs):
            return frozenset((tuple(coef),r) for coef,r in zip(rows,rhs))
        return cls(tuple(context),tuple(family.center),tuple(map(tuple,family.image)),
                   constraints(family.equal,family.equal_rhs),constraints(family.upper,family.upper_rhs))

    def covers(self,other):
        return (self.context==other.context and self.center==other.center and self.image==other.image
                and self.equal<=other.equal and self.upper<=other.upper)


def concave_transport(surface,seam,ball,family,first,second):
    """Fixed-clock concave transport of the same joint input variables.

    Returns None for a convex seam: the caller must use the pressure solver,
    not silently let this affine model cross it. Clearance and continuous
    containment are a separate check even when the conditioned LP is nonempty.
    """
    i,j=seam.patches
    if {i,j}!={first,second}:
        raise ValueError('transport does not match seam')
    a,b=seam.endpoints
    p,q=surface.patches[first],surface.patches[second]
    if first!=i:a,b=b,a
    outward=sc.unit(np.cross(b-a,p.normal))
    if q.normal@outward>EPS:
        return None
    r=ball.radius+sc.SKIN
    return family.on_edge(a+r*p.normal,b+r*p.normal).transfer(p.normal,q,r)
