"""Finite, analytic P_Box transitions. No orientation or time stepping."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from shapely import LineString, Point, union_all

from . import closedform as cf, conformal, motion, physics as ph, props, supportcarry as sc

EPS=sc.EPS
XZ=[0,2]


def feature(point, center, half):
    """Closest exterior cube feature and normal pointing from box to ball."""
    q=np.asarray(point)-center; near=np.clip(q,-half,half); delta=q-near
    if np.linalg.norm(delta)<=EPS:return None
    return center+near,sc.unit(delta),('face','edge','corner')[np.count_nonzero(np.abs(delta)>EPS)-1]


def impact(ball, velocity, box_velocity, normal, spec, *, omega=None, friction=False, supported=False):
    """Normal impulse, with optional Coulomb alternative at the ball contact.

    A supported box may not acquire a downward velocity on a horizontal floor.
    The normal effective mass then uses the horizontal projection on its side.
    Fixed orientation means no inverse rotational mass on the box side.
    """
    v=np.asarray(velocity,float); u=np.asarray(box_velocity,float); n=sc.unit(np.asarray(normal,float))
    spin=np.zeros(3) if omega is None else np.asarray(omega,float)
    r=-ball.radius*n; arm=np.cross(spin,r)
    inv=np.eye(3)/spec['mass']; inertia=sc.inertia_ratio(ball)*ball.mass*ball.radius**2
    if supported and n[1]>0.:inv[1,1]=0.
    vn=float((v-u)@n)
    if vn>=-EPS:return None
    j=-(1+math.sqrt(ball.elasticity*spec['elasticity']))*vn/(1/ball.mass+n@inv@n)
    J=j*n
    if friction:
        w=v-u+arm+J/ball.mass+inv@J; w-=n*(w@n)
        if np.linalg.norm(w)>EPS:
            # Solve the 2-D effective mass, then clamp to the Coulomb disk.
            e=sc.unit(w); f=np.cross(n,e); basis=np.stack((e,f),axis=1)
            K=inv+np.eye(3)*(1/ball.mass+ball.radius**2/inertia)
            t=basis@np.linalg.solve(basis.T@K@basis,-basis.T@w)
            J=J+t*min(1.,ball.friction*spec['friction']*j/np.linalg.norm(t))
    return dict(ball_velocity=v+J/ball.mass,box_velocity=u-inv@J,
                omega=spin+np.cross(r,J)/inertia,impulse=J,normal_impulse=j)


@dataclass(frozen=True)
class Leg:
    p: np.ndarray
    v: np.ndarray
    impulse: np.ndarray
    loss: float
    ticks: float
    kind: str

    def at(self,t):
        dx,v=cf._motion(t,self.v,self.impulse,self.loss)
        return self.p+dx,v

    def curve(self):
        return motion.Piece('air',self.p,self.v,self.impulse,self.loss,self.ticks)


def slide(p,v,mu,spec):
    speed=np.linalg.norm(v)
    if speed<=EPS:return None
    impulse=-mu*ph.GRAVITY*ph.DT*v/speed; loss=spec['lin_damp']*ph.DT
    if mu<=0.:return None
    t=cf._turn(speed,-mu*ph.GRAVITY*ph.DT,loss)
    return Leg(p,v,impulse,loss,t,'slide') if t>EPS else None


def first_exit(poly,p,q):
    """End of the connected footprint interval containing the start."""
    if np.linalg.norm(q-p)<=EPS:return 1.
    line=LineString((p,q)); hit=poly.intersection(line)
    for part in getattr(hit,'geoms',(hit,)):
        if part.geom_type=='LineString' and part.distance(Point(p))<=EPS:
            d=q-p
            return max(float((np.asarray(x)-p)@d/(d@d)) for x in (part.coords[0],part.coords[-1]))
    return 0.


def _material(mesh,patch,key,default):
    return float(mesh.entities[mesh.owner[patch.faces[0]]].get(key,default))


class Floor:
    """Only horizontal supporting charts; unsupported oblique obstacles fail closed."""
    def __init__(self,surface,mesh,spec):
        from shapely import STRtree, box
        self.mesh=mesh; self.spec=spec; self.half=float(spec['half'][0]); self.patches=surface.patches
        self.tri=surface.triangles; self.lo=self.tri.min(1);self.hi=self.tri.max(1)
        self.tree=STRtree([box(*np.r_[a[XZ],b[XZ]]) for a,b in zip(self.lo,self.hi)])
        self.flat=[j for j,p in enumerate(self.patches) if p.normal[1]>0. and abs(np.diff(p.heights())[0])<=conformal.PLANE_ERROR]

    def at(self,c):
        b=c[1]-self.half; foot=Point(c[XZ]); error=conformal.PLANE_ERROR
        found=[j for j in self.flat if abs(self.patches[j].height(c[XZ])-b)<=error and self.patches[j].polygon.distance(foot)<=error]
        return min(found,key=lambda j:abs(self.patches[j].height(c[XZ])-b),default=None)

    def material(self,j,key,default):return _material(self.mesh,self.patches[j],key,default)

    def candidates(self,a,b):
        from shapely import box
        r=self.half
        return self.tree.query(box(*np.r_[np.minimum(a,b)[XZ]-r,np.maximum(a,b)[XZ]+r]))

    def wall(self,a,b):
        """First finite axial wall; unclear feature ordering rejects the move."""
        from shapely import Polygon, box
        d=b-a; i=int(np.argmax(np.abs(d))); jj=[j for j in range(3) if j!=i]; sign=np.sign(d[i]); r=self.half; tol=conformal.PLANE_ERROR
        hits=[]
        for f in self.candidates(a,b):
            if self.hi[f,1]<=a[1]-r+tol or self.lo[f,1]>=a[1]+r-tol:continue
            if np.any(self.hi[f]<np.minimum(a,b)-r+EPS) or np.any(self.lo[f]>np.maximum(a,b)+r-EPS):continue
            lead=(self.lo[f,i] if sign>0 else self.hi[f,i])-sign*r
            s=(lead-a[i])/d[i]
            if s>1.+EPS:continue
            if self.hi[f,i]-self.lo[f,i]<=EPS:
                if s < -tol/abs(d[i]):return 0.,None
                q=a[jj]; polygon=Polygon(self.tri[f][:,jj]); square=box(*(np.r_[q-r,q+r]))
                if polygon.intersection(square).area<=EPS:continue
                hits.append((max(0.,s),int(f)))
            else:hits.append((max(0.,s),None))
        return min(hits,key=lambda x:(x[0],x[1] is not None)) if hits else (math.inf,None)

    def move(self,c,v,j):
        spec=self.spec; r=self.half; tol=conformal.PLANE_ERROR
        p=self.patches[j]; base=p.height(c[XZ]); fr=self.material(j,'friction',.7)
        same=union_all([self.patches[k].polygon for k in self.flat if abs(self.patches[k].height(c[XZ])-base)<=tol
                        and self.material(k,'friction',.7)==fr and self.patches[k].polygon.distance(Point(c[XZ]))<=
                        np.linalg.norm(v)**2/(2*spec['friction']*fr*ph.GRAVITY)+r])
        pieces=[]; wall_count=0
        for _ in range(3):  # search budget; an unresolved second rebound is not a rest state
            leg=slide(c,v,spec['friction']*fr,spec)
            if leg is None:return (c,pieces,'wall_rest' if wall_count else 'floor_rest')
            end=leg.at(leg.ticks)[0]
            s=first_exit(same,c[XZ],end[XZ]); w,f=self.wall(c,c+(end-c)*min(1.,s)) if s>EPS else (math.inf,None)
            if w<=1.+EPS:
                if f is None:return None
                target=min(1.,s)*w
                why='wall'
            else:target=s;why='ledge' if s<1.-EPS else 'rest'
            t=leg.ticks if target>=1.-EPS else 0. if target<=EPS else brentq(
                lambda x: ((leg.at(x)[0]-c)@(end-c))/((end-c)@(end-c))-target,0.,leg.ticks)
            piece=Leg(c,v,leg.impulse,leg.loss,t,'slide'); pieces.append(piece);c,v=piece.at(t)
            if why=='rest':return c,pieces,'floor_rest' if not wall_count else 'wall_rebound_rest'
            if why=='ledge':
                fall=self.fall(c,v,base)
                if fall is None:return None
                c,lower,air=fall;pieces.append(air)
                landing=self.lower_bounces(c,air.at(air.ticks)[1],lower)
                if landing is None:return None
                c,v,small_hops=landing; pieces.extend(small_hops)
                if slide(c,v,spec['friction']*self.material(lower,'friction',.7),spec) is None:return c,pieces,'lower_rest'
                rest=self.move(c,v,lower) if self.patches[lower].height(c[XZ])<base-tol else None
                return (rest[0],pieces+rest[1],'lower_'+rest[2]) if rest is not None else None
            wall_count+=1
            if np.linalg.norm(v)<=ph.Params().bounce_speed:return c,pieces,'wall_rest'
            e=math.sqrt(spec['elasticity']*float(self.mesh.entities[self.mesh.owner[f]].get('elasticity',.3)))
            if t<=EPS:return None
            v=-e*v
        return None

    def lower_bounces(self,c,v,j):
        """Vertical rebound off a checked lower plane; zero normal speed at capture."""
        r=self.half; imp=np.array([0.,-ph.GRAVITY*ph.DT,0.]); loss=self.spec['lin_damp']*ph.DT
        p=self.patches[j]; e=math.sqrt(self.spec['elasticity']*self.material(j,'elasticity',.3)); legs=[]
        for _ in range(8):  # search budget; no fabricated rest pose on exhaustion
            if -v[1]<=ph.Params().bounce_speed:
                v=v.copy(); v[1]=0.; return c,v,legs
            u=v.copy(); u[1]*=-e
            turn=cf._turn(u[1],imp[1],loss)
            if turn<=EPS:return None
            t=cf._increasing_root(-u[1],-imp[1],loss,0.,turn)
            if t is None:return None
            leg=Leg(c,u,imp,loss,t,'lower_rebound'); q,w=leg.at(t)
            if not p.polygon.covers(LineString((c[XZ],q[XZ]))) and np.linalg.norm(q[XZ]-c[XZ])>EPS:return None
            clip=conformal.PLANE_ERROR
            for f in self.candidates(c,q):
                if self.hi[f,1]<=c[1]-r+clip:continue
                if motion.box_hit(leg.curve(),self.lo[f]-r,self.hi[f]+r) is not None:return None
            legs.append(leg); c,v=q,w
        return None

    def fall(self,c,v,upper):
        """First descending horizontal bottom contact; other features veto it."""
        r=self.half; loss=self.spec['lin_damp']*ph.DT; imp=np.array([0.,-ph.GRAVITY*ph.DT,0.]); tol=conformal.PLANE_ERROR
        def at(t):return c+cf._motion(t,v,imp,loss)[0]
        hits=[]
        for j in self.flat:
            p=self.patches[j]; y=p.height(c[XZ]); drop=upper-y
            if drop<=tol:continue
            # A terminal-speed bound plus one decay time brackets the unique descending root.
            top=2/loss+drop*loss/(ph.GRAVITY*ph.DT**2)
            t=brentq(lambda n:at(n)[1]-r-y,0.,top); q=at(t)
            if p.polygon.covers(Point(q[XZ])):hits.append((t,j,q))
        if not hits:return None
        t,j,q=min(hits,key=lambda x:(x[0],x[1])); leg=Leg(c,v,imp,loss,t,'fall')
        for f in self.candidates(c,q):
            if self.hi[f,1]<=q[1]-r+tol:continue
            # The releasing horizontal floor has no support once the box centre leaves.
            if abs(self.hi[f,1]-upper)<=tol and abs(self.lo[f,1]-upper)<=tol:continue
            if motion.box_hit(leg.curve(),self.lo[f]-r,self.hi[f]+r) is not None:return None
        return q,j,leg

def _point_rate(t,v,b,loss):
    terminal=b/loss
    return ph.DT*(terminal+(v-terminal)*(-math.log1p(-loss)/loss)*math.exp(t*math.log1p(-loss)))


def _relative_clear(ball_legs,box_legs,d):
    """Prove the horizontal separating gap never becomes negative.

    Between analytic event boundaries the rate is a constant plus two
    exponentials. Its sole possible extremum isolates all displacement roots.
    """
    offsets=lambda legs:np.r_[0.,np.cumsum([p.ticks for p in legs])]
    starts=[offsets(x) for x in (ball_legs,box_legs)]
    end=max(s[-1] for s in starts); marks=sorted({0.,end,*starts[0],*starts[1]})
    seq=(ball_legs,box_legs)
    original=tuple(float(ls[0].p@d) for ls in seq)
    for a,b in zip(marks,marks[1:]):
        active=[]
        for legs,s in zip(seq,starts):
            j=int(np.searchsorted(s,(a+b)/2,side='right'))-1
            active.append((legs[j],s[j],True) if j<len(legs) else (legs[-1],s[-2],False))
        coeff=[]
        for leg,t0,live in active:
            L=math.log1p(-leg.loss); v=float(leg.v@d); impulse=float(leg.impulse@d)
            coeff.append(((v-impulse/leg.loss)*(-L/leg.loss)*L*math.exp((a-t0)*L) if live else 0.,L))
        def gap(t):
            x,y=[float(leg.at(t-t0 if live else leg.ticks)[0]@d)-p for (leg,t0,live),p in zip(active,original)]
            return y-x
        def rate(t):
            x,y=[_point_rate(t-t0,float(leg.v@d),float(leg.impulse@d),leg.loss) if live else 0. for leg,t0,live in active]
            return y-x
        (A,L),(B,K)=coeff; roots=[a,b]
        if A*B>0. and abs(L-K)>EPS:
            t=a+math.log(B/A)/(L-K)
            if a<t<b:roots.insert(1,t)
        candidates=list(roots)
        for l,h in zip(roots,roots[1:]):
            if rate(l)*rate(h)<0.:candidates.append(brentq(rate,l,h))
        if min(map(gap,candidates)) < -EPS:return False
    return True


def _overlaps(lo,hi,q,r):
    return bool(np.all(hi>q-r+EPS) and np.all(lo<q+r-EPS))


def _bound(legs,r):
    a,b=zip(*(x.curve().bounds(0.,x.ticks) for x in legs))
    return np.min(a,axis=0)-r,np.max(b,axis=0)+r


def proposal(floor,g,body,j,ball,key,d,other):
    """A centred face impact from rest with a literal forward and brake key."""
    c=np.asarray(body['rest']);p=floor.patches[j];r=ball.radius+sc.SKIN; h=floor.half
    at=c-(h+r)*d; at[1]=p.height(at[XZ],r)
    ranges=g.info['standing_ranges'][ball.name]; area=ranges.get(j)
    if area is None or not area.covers(Point(at[XZ])):return None
    bounds=np.asarray(area.bounds).reshape(2,2); limit=float(np.linalg.norm(bounds[1]-bounds[0]));
    available=limit*first_exit(area,at[XZ],at[XZ]-limit*d[XZ])
    if available<=EPS:return None
    axis=sc.law(ball,sc.UP,key).along(d); span=axis.over(0.,available/2)
    if span is None:return None
    speed,sec=span; t=sec/ph.DT; origin=at-available/2*d
    hit=impact(ball,speed*d,np.zeros(3),-d,floor.spec,friction=False,supported=True)
    if hit is None:return None
    moved=floor.move(c,hit['box_velocity'],j)
    if moved is None:return None
    end,legs,reason=moved
    if not legs or np.linalg.norm(end-c)<=conformal.PLANE_ERROR:return None
    wait=math.ceil(t)-t; step=Leg(at,hit['ball_velocity'],axis.impulse*d,axis.loss,wait,'hold')
    apex,v=step.at(wait); stop_axis=sc.law(ball,sc.UP,-key).along(d)
    turn=cf._turn(float(v@d),stop_axis.impulse,stop_axis.loss)
    stop=Leg(apex,v,stop_axis.impulse*d,stop_axis.loss,float(math.ceil(turn)),'brake')
    brake_end,velocity=stop.at(stop.ticks)
    if velocity@d>EPS:return None
    player=[step,stop]; coast_duration=max(0.,sum(p.ticks for p in legs)-sum(p.ticks for p in player))
    coast=Leg(brake_end,velocity,np.zeros(3),axis.loss,coast_duration,'coast')
    q=coast.at(coast_duration)[0]
    far=brake_end+ph.DT*velocity/axis.loss
    near=stop.at(turn)[0]
    if not area.covers(LineString((at[XZ],near[XZ],far[XZ]))):return None
    if coast_duration>EPS:player.append(coast)
    if not _relative_clear(player,legs,d):return None
    runup=Leg(origin,np.zeros(3),axis.impulse*d,axis.loss,t,'runup'); player=(runup,*player)
    bb=_bound(legs,h); pb=_bound(player,r)
    pb=(np.minimum(pb[0],far-r),np.maximum(pb[1],far+r))
    if any(_overlaps(*bound,where,h) for bound in (bb,pb) for where in other):return None
    keys=[(math.ceil(t),key),(math.ceil(t)+int(stop.ticks),-key)]
    total=t+sum(p.ticks for p in player[1:]); last=math.ceil(total)
    if last>keys[-1][0]:keys.append((last,np.zeros(2)))
    return dict(ball=ball.name,patch=j,box_start=c,box_end=end,end_reason=reason,ticks=total,
        box_motion=legs,ball_motion=player,ball_end=q,impact=hit,keys=tuple(keys),bounds=(bb,pb))


def state_config(g,mesh,bodies,placements,sector):
    """One proved destination per box; independently composable state bits."""
    ps={p['name']:p for p in placements if p['type']=='P_Box' and p.get('sector')==sector}
    live=[b for b in bodies if b['name'] in ps and not b.get('gone')]
    if not live:return []
    floor=Floor(g.info['conformal'],mesh,props.bodies()['box']); out=[]
    where={b['name']:np.asarray(b['rest']) for b in live}
    for body in live:
        name=body['name'];p=ps[name]; c=where[name]
        j=floor.at(c); attempts=[]
        aligned=all(np.allclose(p.get(k+'_axis',e),e,atol=EPS,rtol=0.) for k,e in zip('xyz',np.eye(3)))
        others=[u for k,u in where.items() if k!=name]+[m['push']['box_end'] for m in out if m['push'] is not None]
        if j is not None and aligned and body.get('settled',True):
            for d in (np.eye(3)[0],-np.eye(3)[0],np.eye(3)[2],-np.eye(3)[2]):
                for ball in ph.BALLS.values():
                    q=proposal(floor,g,body,j,ball,d[XZ],d,others)
                    if q is not None and not any(_overlaps(*z,q['box_end'],floor.half) for m in out if m['push'] is not None for z in m['push']['bounds']):
                        attempts.append(q)
        push=max(attempts,key=lambda a:(a['end_reason'].startswith('lower'),np.linalg.norm(a['box_end']-c)),default=None)
        out.append(dict(name=name,type='loose_box',origin=c,half=floor.half if aligned else floor.half*math.sqrt(3.),push=push,edges=[],
                        status='verified_push' if push else 'no_certified_push' if aligned else 'non_axis_aligned'))
    return out


def intersects(path,center,half,r):
    lo=center-half-r;hi=center+half+r
    for p in path:
        if p.kind=='line':
            if motion.line_box_distance(p.p-center,p.v,np.full(3,half),0.,p.end)<r-EPS:return True
        elif motion.box_hit(p,lo,hi) is not None:return True  # conservative rounded-cube outer bound
    return False
