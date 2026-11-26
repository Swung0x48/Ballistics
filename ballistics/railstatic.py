"""Finite quasi-static rail certificates, independent of coarse stops and keys."""
from collections import Counter, defaultdict
import math
from types import SimpleNamespace

import numpy as np
from shapely import LineString, Point, STRtree, box

from . import closedform as cf, conformal, motion, physics as ph, quasistatic, railinlet, railride, rails, rimclear, supportcarry as sc, sweep, terrain

XZ = [0, 2]
EPS = sc.EPS
TOL = conformal.PLANE_ERROR
SOURCE = 'quasistatic_rail'


def axial_margin(ball, t):
    """A horizontal cancellation of axial gravity with lateral reserve.

    For a chosen axial direction t, F_h = g*t_y*t_h/|t_h|^2 has no
    transverse component. An open force box about it gives a nonzero
    interval for forward, backward and transverse corrections.
    """
    t=np.asarray(t)
    q=float(t[XZ]@t[XZ])
    if q<=EPS**2:return -math.inf
    return terrain.bounds(ball)[0]-float(np.max(np.abs(t[1]*t[XZ]/q)))


def covers(intervals):
    end=0.
    for lo,hi in sorted(intervals):
        if lo>end+EPS:return False
        end=max(end,hi)
        if end>=1.-EPS:return True
    return False


def segment_features(tf, obs, a,b,r, *, full_intervals=False):
    """Clip raw feature cells against the contact band, using linear/quadratic roots."""
    d=b-a; hi=r+TOL
    ids=np.asarray(tf._box_triangles((np.minimum(a,b)-hi)[None],(np.maximum(a,b)+hi)[None],obs.lo,obs.hi,obs.mesh._grid),dtype=int)
    if not len(ids):return [],[] if full_intervals else False
    normals=obs.mesh.N[ids]; tri=obs.tris[ids]
    dist=np.einsum('ij,ij->i',a-tri[:,0],normals)
    last=dist+normals@d
    ids=ids[np.maximum(np.minimum(dist,last),-np.maximum(dist,last))<=hi]
    from .geometry import _closest
    mid=(a+b)*.5; reach=(hi+.5*np.linalg.norm(d)+EPS)**2
    rows=[]; blocked=[]
    for f in ids:
        tri=obs.tris[f]; n=obs.mesh.N[f]; delta=mid-_closest(*mid,*tri.ravel())[:3]
        if delta@delta>reach:continue
        for k in range(7):
            l,h,w,v=sweep.feature_span(tri,n,k,a,d)
            l,h=sweep.sphere_span(w,v,hi,l,h)
            if l>h:continue
            x,y=sweep.sphere_span(w,v,r-TOL-EPS,l,h)
            if x<=y:
                if not full_intervals:return [],True
                blocked.append((x,y))
            if obs.mesh.rail[f]:rows.append((int(f),l,h,w,v))
    return rows,blocked if full_intervals else False


def bearing_windows(rows,t,r,gauge,tube,margin):
    """Support intervals; a pair requires distinct signed flange reactions.

    A radial with too much longitudinal component belongs to an end cap,
    not the straight chart. For a crown, lateral reaction must fit the
    reserved horizontal correction at the same station.
    """
    up=railride.support_normal(t); side=sc.unit(np.cross(up,t)); pair=gauge>0.
    bands=[[],[]] if pair else [[]]
    arm=(gauge*.5*r/(r+tube)-TOL) if pair else 0.
    for f,l,h,w,v in rows:
        l,h=sweep.linear_clip(float(w@up)-EPS,float(v@up),l,h)
        for s in (-1.,1.):
            l,h=sweep.linear_clip(TOL+s*float(w@t),s*float(v@t),l,h)
        if h<l:continue
        if pair:
            # Opposite normals give positive coefficients for a vertical
            # cross-section resultant; both have positive normal load.
            for i,s in enumerate((-1.,1.)):
                x,y=sweep.linear_clip(s*float(w@side)-max(EPS,arm),s*float(v@side),l,h)
                if x<=y:bands[i].append((x,y))
        else:
            for s in (-1.,1.):
                axis=margin*up+s*side
                l,h=sweep.linear_clip(float(w@axis),float(v@axis),l,h)
            if l<=h:bands[0].append((l,h))
    if len(bands)==1:return merge(bands[0])
    a,b=map(merge,bands); out=[];i=j=0
    while i<len(a) and j<len(b):
        l=max(a[i][0],b[j][0]);h=min(a[i][1],b[j][1])
        if h-l>EPS:out.append((l,h))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def merge(windows):
    out=[]
    for a,b in sorted(windows):
        if out and a<=out[-1][1]+EPS:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out


def unblocked(windows,blocked):
    stops=merge(blocked); result=[]
    for l,h in windows:
        for a,b in stops:
            if b<l:continue
            if a>h:break
            # Preserve a numerical gap at a penetrated feature root.
            if a-l>2*EPS:result.append((l,a-EPS))
            l=max(l,b+EPS)
        if h-l>EPS:result.append((l,h))
    return result


def bearing_intervals(rows,t,r,gauge,tube,margin):
    return covers(bearing_windows(rows,t,r,gauge,tube,margin))


def _charts(lane,ball):
    if lane.axis is None:
        return [(a,b) for a,b in zip(lane.p,lane.p[1:])]
    h=ball.radius+sc.SKIN+lane.tube
    return [(a+h*railride.support_normal(t),b+h*railride.support_normal(t))
            for a,b,t in zip(lane.axis,lane.axis[1:],lane.tangent)]


def corrected_charts(paths,lane,ball):
    """Propose a transverse (not world-vertical) offset on finite tube axes."""
    r=ball.radius+sc.SKIN; R=r+lane.tube
    if lane.axis is not None:return _charts(lane,ball)
    if not lane.double:
        return [(a+(R*railride.support_normal(t)-(ph.WOOD.radius+lane.tube)*sc.UP),
                 b+(R*railride.support_normal(t)-(ph.WOOD.radius+lane.tube)*sc.UP))
                for a,b,t in zip(lane.p,lane.p[1:],lane.tangent)]
    pair=paths.pairs.get(lane.parent)
    if pair is None:return _charts(lane,ball)
    x,y=(paths.lanes[j] for j in pair); out=[]
    for a,b,t in zip(lane.p,lane.p[1:],lane.tangent):
        mid=(a+b)*.5; ends=[]
        for p in (a,b):
            # The perpendicular sections at the two ring stations meet
            # finite axes. Solve their station parameters, not a point-grid
            # nearest-distance compromise between different angles.
            bases=[]
            for axis in (x,y):
                j,_=rails.project_pieces(axis.axis,mid); u=axis.tangent[j]
                turn=float(u@t)
                if abs(turn)<=EPS:break
                z=float((p-axis.axis[j])@t/turn)
                if not -TOL<=z<=axis.length[j]+TOL:break
                bases.append(axis.axis[j]+np.clip(z,0.,axis.length[j])*u)
            if len(bases)!=2:break
            w=bases[1]-bases[0]; half=np.linalg.norm(w)*.5
            if not EPS<half<R:break
            up=sc.unit(np.cross(t,w));up*=1 if up[1]>0. else -1
            if up[1]<=EPS:break
            ends.append((bases[0]+bases[1])*.5+math.sqrt(R*R-half*half)*up)
        out.append(tuple(ends) if len(ends)==2 else (a,b))
    return out


def _new_chart_nodes(g,paths):
    """A crown is not an alias for the groove at the same along-distance."""
    extra=[]; ids={}; roles={}
    for i,lane in paths.lanes.items():
        if lane.axis is None:
            ids[i]=g.info['rail_piece_ids'][lane.parent][:len(lane.length)]
        else:
            ids[i]=tuple(range(len(g.pos)+len(extra),len(g.pos)+len(extra)+len(lane.length)))
            extra.extend((a+b)*.5 for a,b in _charts(lane,ph.WOOD))
        roles.update({u:(lane.parent,lane.line,j) for j,u in enumerate(ids[i])})
    if extra:
        g.pos=np.vstack((g.pos,extra));g.adj.extend([] for _ in extra)
        g.kind=np.r_[g.kind,np.ones(len(extra),dtype=g.kind.dtype)]
        # Standalone route-chart nodes must not enter _lane_chains, which
        # reads the original centre-line fitting, not crown identities.
        g.lane=np.r_[g.lane,np.full(len(extra),-1,dtype=g.lane.dtype)]
        g.info['edge_node']=np.r_[g.info['edge_node'],np.ones(len(extra),dtype=bool)]
    g.info['rail_chart_nodes']=roles
    return ids


def miter_station(a,b,c,d):
    """An exact shared station in the two certified finite center charts."""
    if np.linalg.norm(b-c)<=EPS:return b
    u=b-a;v=d-c; matrix=np.column_stack((u,-v))
    if np.linalg.norm(np.cross(u,v))<=EPS:return None
    s,t=np.linalg.lstsq(matrix,c-a,rcond=None)[0]
    if not .5-EPS<=s<=1.+EPS or not -EPS<=t<=.5+EPS:return None
    x,y=a+s*u,c+t*v
    return (x+y)*.5 if np.linalg.norm(x-y)<=EPS else None


def norm_roots(D,r,hi):
    """Roots of |c+a cos+b sin|^2=r^2 on [0,hi), hi < pi."""
    c,a,b=D
    terms=np.stack((c+a,2*b,c-a))
    poly=np.convolve(terms[:,0],terms[:,0])+np.convolve(terms[:,1],terms[:,1])+np.convolve(terms[:,2],terms[:,2])
    poly-=r*r*np.array([1.,0.,2.,0.,1.])
    return tuple(2*math.atan(x.real) for x in np.polynomial.polynomial.polyroots(poly)
                 if abs(x.imag)<=EPS and 0.<2*math.atan(x.real)<hi)


def crown_arc(tf,obs,ball,axis,t,tube,n,m,l,h):
    """Contact-band coverage for a fixed-station quasi-static crown arc."""
    if not quasistatic.holds(ball,n) or not quasistatic.holds(ball,m):return None
    dot=float(n@m); angle=math.acos(np.clip(dot,-1.,1.)); r=ball.radius+sc.SKIN
    if not EPS<angle<math.pi/2:return None
    b=sc.unit(m-dot*n); R=r+tube
    curve=motion.arc(axis,R*n,R*b,0.,angle,np.zeros(3),(1.,0.,0.))
    if any(motion.box_hit(curve,x,y) is not None for x,y in obs.boxes):return None
    low,high=curve.bounds(0.,1.);pad=r+TOL
    fs=np.asarray(tf._box_triangles((low-pad)[None],(high+pad)[None],obs.lo,obs.hi,obs.mesh._grid),dtype=int)
    normals=obs.mesh.N[fs]; lo,hi=curve.bounds(0.,1.,normals)
    off=np.einsum('ij,ij->i',normals,obs.tris[fs,0]); fs=fs[np.maximum(lo-off,off-hi)<=pad]
    C=np.stack((axis,curve.v,curve.b)); inn= r-TOL-EPS; span=[]
    left=np.eye(3)-np.outer(t,t)
    for f in fs:
        tri=obs.tris[f]
        for kind,ids,projection,H,Y,_ in rimclear.features(tri):
            limits=H@C.T;limits[:,0]-=Y
            D=(C-np.stack((tri[ids[0]],np.zeros(3),np.zeros(3))))@(np.eye(3)-projection)
            # Raw contact point's radial distance to the actual tube axis,
            # and its finite axial parameter, are checked independently.
            Q=C-D; Q[0]-=axis; tube_delta=Q@left
            station=Q@t
            test=np.vstack((limits,station-[h,0.,0.],[l,0.,0.]-station,-D@railride.support_normal(t)))
            cuts={0.,angle,*norm_roots(D,inn,angle),*norm_roots(D,pad,angle)}
            for row in test:cuts.update(rimclear.trig_zeros(*row,0.,angle))
            if obs.mesh.rail[f]:
                cuts.update(norm_roots(tube_delta,max(0.,tube-TOL),angle));cuts.update(norm_roots(tube_delta,tube+TOL,angle))
            cuts=sorted(cuts)
            for x,y in zip(cuts,cuts[1:]):
                z=(x+y)*.5; basis=np.array([1.,math.cos(z),math.sin(z)])
                if np.any(limits@basis>EPS):continue
                dist=np.linalg.norm(basis@D)
                if dist<inn:return None
                if obs.mesh.rail[f] and dist<=pad+EPS and np.all(test@basis<=EPS) and abs(np.linalg.norm(basis@tube_delta)-tube)<=TOL+EPS:
                    span.append((x/angle,y/angle))
    return curve if covers(span) else None


def _road_leg(tf,obs,patch,p,center,r,t):
    """Finite points on the offset plane, not patch display representatives."""
    n=patch.normal
    for d in (t,-t):
        d=sc.unit(d-(d@n)*n)
        depth=railinlet._approach(patch,p-r*n,d)
        if depth<=EPS:continue
        at=p-min(depth*.5,r)*d
        if any(motion.box_hit(motion.line(a,b),lo,hi) is not None for a,b in ((at,p),(p,center)) for lo,hi in obs.boxes):continue
        if all(not segment_features(tf,obs,a,b,r)[1] for a,b in ((at,p),(p,center))):return at
    return None


def certify(tf,g,obs,layout,add,publish):
    from . import railtransition
    surface=g.info['conformal'];paths=railride.RailPaths(g,obs,SimpleNamespace(near=lambda f,_:int(surface.face_patch[f])))
    ids=_new_chart_nodes(g,paths); stats=Counter(); rows=[]; edges=[]; exits=[]
    ends=defaultdict(list); joints=[]; portals=[]
    polys=[p.polygon for p in surface.patches];tree=STRtree(polys)
    # Geometry is identical between balls of the same radius; evaluate
    # each finite feature intersection once, not once per ball/direction.
    groups=defaultdict(list)
    for ball in ph.BALLS.values():groups[ball.radius].append(ball)
    roles=g.info['rail_chart_nodes']
    def link(a,b,ball,rule,spans):
        if any(u in roles and roles[u][1]!='groove' for u in (a,b)):
            stats['single_crown_requires_dynamic']+=1
            return
        pieces=tuple(piece if isinstance(piece,motion.Piece) else motion.line(*piece)
                     for piece in spans if isinstance(piece,motion.Piece) or np.linalg.norm(piece[0]-piece[1])>EPS)
        if not pieces:return
        bounds=(pieces[0].point(0.),pieces[-1].point(pieces[-1].end)); add(a,b,ball.name,SOURCE,bounds,kind='rail',motion=pieces)
        stats[rule]+=1;edges.append(dict(source=a,target=b,ball=ball.name,rule=rule,span=bounds))
    for i,lane in paths.lanes.items():
        nodes=ids[i]; gauge=lane.gauge if lane.double else 0.
        for radius,balls in groups.items():
            r=radius+sc.SKIN; chart=_charts(lane,balls[0]); data=[]
            alternative=corrected_charts(paths,lane,balls[0]); corrected=set()
            def geometry(a,b):
                dead=any(motion.box_hit(motion.line(a,b),lo,hi) is not None for lo,hi in obs.boxes)
                cells,hit=([],[]) if dead else segment_features(tf,obs,a,b,r,full_intervals=True)
                return dead,hit,cells
            for j,(a,b) in enumerate(chart):
                dead,hit,cells=geometry(a,b); t=sc.unit(b-a)
                best=max(axial_margin(ball,t) for ball in balls)
                alt=alternative[j]; difer=np.linalg.norm(np.asarray(alt)-np.asarray((a,b)))
                if difer>EPS and (dead or hit or best<=EPS or not bearing_intervals(cells,t,r,gauge,lane.tube,best)):
                    other_block,other_hit,new=geometry(*alt); u=sc.unit(alt[1]-alt[0]); reserve=max(axial_margin(ball,u) for ball in balls)
                    if not (other_block or other_hit) and reserve>EPS and bearing_intervals(new,u,r,gauge,lane.tube,reserve):
                        chart[j]=alt;dead,hit,cells=other_block,other_hit,new;corrected.add(j)
                data.append((dead,hit,cells))
            for ball in balls:
                verdict=[]; safe=[]; windows=[]
                for j,((a,b),(dead,hit,cells)) in enumerate(zip(chart,data)):
                    t=sc.unit(b-a); margin=axial_margin(ball,t)
                    all_spans=unblocked(bearing_windows(cells,t,r,gauge,lane.tube,margin),hit) if margin>EPS and not dead else []
                    span=max(all_spans,key=lambda e:(e[0]<=.5<=e[1],e[1]-e[0]),default=None)
                    reason=('supported' if span is not None else 'death_box' if dead else 'raw_mesh_penetration' if hit else
                            'downhill_momentum_required' if margin<=EPS else 'incomplete_raw_bearing')
                    windows.append(span); safe.append(tuple(a+x*(b-a) for x in span) if span is not None else (a,b))
                    if span is not None and (span[0]>EPS or span[1]<1.-EPS):stats['piece_partial_support']+=1
                    stats['piece_'+reason]+=1;verdict.append(reason)
                    rows.append(dict(chain=lane.parent,line=lane.line,piece=j,ball=ball.name,grade=float(t[1]/max(EPS,np.linalg.norm(t[XZ]))),
                                     force_margin=margin if math.isfinite(margin) else None,verdict=reason,span=safe[-1],
                                     original_span=(a,b),certified_abscissa_range=span,
                                     offset_chart='transverse_tube_section' if j in corrected else 'conformal_rail',
                                     forward='downhill_momentum' if margin<=EPS and t[1]<0 else 'uphill_requires_entry_family' if margin<=EPS else reason,
                                     reverse='downhill_momentum' if margin<=EPS and t[1]>0 else 'uphill_requires_entry_family' if margin<=EPS else reason))
                for j in range(len(chart)-1):
                    a,b=safe[j];c,d=safe[j+1]; reason='adjacent_piece_unproved'
                    if verdict[j]==verdict[j+1]=='supported':
                        position=miter_station(a,b,c,d)
                        # The fitted v33 miter can carry a compressive change
                        # of axial direction. A noncoincident crown, however,
                        # needs a supported continuous centre locus; never
                        # insert a chord across the outside of that pipe.
                        if (b-a)@(d-c)<=EPS:
                            reason='foldback_requires_continuation'
                        elif position is None:
                            reason='offset_miter_requires_continuation'
                            if np.linalg.norm(c-b)<=TOL:
                                direction=sc.unit(sc.unit(b-a)+sc.unit(d-c))
                                cells,hit=segment_features(tf,obs,b,c,r)
                                dead=any(motion.box_hit(motion.line(b,c),x,y) is not None for x,y in obs.boxes)
                                reserve=axial_margin(ball,direction)
                                if not (hit or dead) and reserve>EPS and bearing_intervals(cells,direction,r,gauge,lane.tube,reserve):
                                    reason='raw_supported_offset_miter'
                                    parts=(((a+b)*.5,b),(b,c),(c,(c+d)*.5))
                                    link(nodes[j],nodes[j+1],ball,reason,parts)
                                    link(nodes[j+1],nodes[j],ball,reason,railtransition.reverse_parts(parts))
                        else:
                            reason='coincident_finite_miter'
                            parts=(((a+b)*.5,position),(position,(c+d)*.5))
                            link(nodes[j],nodes[j+1],ball,reason,parts)
                            link(nodes[j+1],nodes[j],ball,reason,tuple((y,x) for x,y in reversed(parts)))
                    stats['joint_'+reason]+=1
                    joints.append(dict(chain=lane.parent,line=lane.line,ball=ball.name,pieces=(j,j+1),
                                       turn_degrees=float(np.degrees(np.arccos(np.clip(lane.tangent[j]@lane.tangent[j+1],-1,1)))),
                                       verdict=reason,offset_gap=float(np.linalg.norm(c-b))))
                # Only regions actually covered by positive normal load
                # and complete straight support can use these entrances.
                for j in range(len(chart)):
                    end_piece=j in (0,len(chart)-1)
                    old_a,old_b=chart[j]; lp=LineString((old_a[XZ],old_b[XZ])); pad=r+TOL
                    close=[]
                    if end_piece:
                        lo=np.minimum(old_a,old_b); hi=np.maximum(old_a,old_b)
                        for k in tree.query(box(*np.r_[lo[XZ]-pad,hi[XZ]+pad])):
                            p=surface.patches[k]
                            if p.polygon.distance(lp)>pad or lo[1]>p.high[1]+2*pad or hi[1]<p.low[1]-pad:continue
                            close.append(int(k))
                    if verdict[j]!='supported':
                        for k in close:
                            portals.append(dict(chain=lane.parent,line=lane.line,piece=j,patch=k,ball=ball.name,
                                                verdict='no_certified_rail_interval',rail_reason=verdict[j]))
                        continue
                    a,b=safe[j]; length=np.linalg.norm(b-a); t=(b-a)/length
                    for s,end in ((-1,a),(1,b)):
                        if (s<0 and (j!=0 or windows[j][0]>EPS)) or (s>0 and (j!=len(chart)-1 or windows[j][1]<1.-EPS)):continue
                        ends[ball.name].append((i,nodes[j],s*t,end,(a+b)*.5,lane.line))
                        state=ph.State(end,np.zeros(3)); push=np.zeros(2)
                        horizon=cf.landing_step(ball,state,min(obs.bottom-r,end[1]-r))
                        if not horizon:continue
                        why,face,tick,where=obs.first_contact(ball,state,push,horizon,incoming_only=True)
                        k=int(surface.face_patch[face]) if face>=0 else -1
                        trace=None
                        if why=='land' and k>=0 and tick>EPS:
                            trace=tf.trace_air(obs,ball,state,((float(horizon),push),),incoming_only=True)
                            why,face,tick,where=trace.result
                            k=int(surface.face_patch[face]) if face>=0 else -1
                            if trace.bounces:why='rebound_requires_dynamic_witness'
                        tag='zero_release_'+('road' if why=='land' and k>=0 and tick>EPS else why if why!='land' else 'rail_or_zero_contact')
                        stats[tag]+=1;exits.append(dict(chain=lane.parent,line=lane.line,piece=j,ball=ball.name,
                            node=nodes[j],position=end,reason=tag,face=face,landed=where))
                        if tag=='zero_release_road' and lane.double:
                            row=dict(origin=(a+b)*.5,launch=end,position=where,face=face,source_patch=-1,target_patch=k,
                                     velocity=np.zeros(3),ticks=tick,flight_ticks=tick,
                                     contacts=trace.contacts,
                                     motion=(motion.line((a+b)*.5,end),*(tf.air_geometry(ball,x,key,t) for x,key,t in trace.parts)))
                            publish(nodes[j],k,ball.name,row,SOURCE,'quasistatic_rail_drop')
                    pad=r+TOL; near=tree.query(box(*np.r_[np.minimum(a,b)[XZ]-pad,np.maximum(a,b)[XZ]+pad]))
                    for k in sorted(set(map(int,near))|set(close)):
                        p=surface.patches[k]; road=layout.patch_nodes[ball.name][k]
                        record=dict(chain=lane.parent,line=lane.line,piece=j,patch=k,ball=ball.name)
                        accepted=False
                        miss='no_common_offset_station'
                        if k not in layout.roads[road].rest_patches or not quasistatic.holds(ball,p.normal):
                            if k in close:portals.append(dict(record,verdict='road_cannot_hold'))
                            continue
                        for l,h in railinlet.bearing_intervals(p,a,t,length,r,TOL):
                            e=float(a@p.normal-p.offset-r); slope=float(t@p.normal)
                            s=float(np.clip(-e/slope,l,h)) if abs(slope)>EPS else (l+h)*.5
                            at=a+s*t; foot=at-(at@p.normal-p.offset-r)*p.normal
                            # Offset mismatch can suggest but cannot
                            # certify an unsupported hop in either direction.
                            if np.linalg.norm(at-foot)>EPS:
                                overlap,hit=segment_features(tf,obs,foot,at,r)
                                if hit or not bearing_intervals(overlap,t,r,gauge,lane.tube,axial_margin(ball,t)):
                                    stats['road_join_offset_unproved']+=1;miss='offset_connector_not_supported';continue
                            x=_road_leg(tf,obs,p,foot,at,r,t)
                            if x is None:stats['road_join_obstructed']+=1;miss='road_approach_obstructed';continue
                            parts=((x,at),(at,(a+b)*.5))
                            link(k,nodes[j],ball,'road_to_rail',parts)
                            link(nodes[j],k,ball,'rail_to_road',tuple((y,z) for z,y in reversed(parts)))
                            accepted=True;portals.append(dict(record,verdict='supported_offset_join'))
                        if k in close and not accepted:
                            parts,why=railtransition.candidates(tf,obs,ball,p,lane,j,chart[j],safe[j])
                            if parts is not None:
                                link(k,nodes[j],ball,'road_lip_to_rail',parts)
                                link(nodes[j],k,ball,'rail_to_road_lip',railtransition.reverse_parts(parts))
                                accepted=True;portals.append(dict(record,verdict=why))
                            elif why!='no_lip_cylinder_root' or miss=='no_common_offset_station':miss=why
                        if lane.axis is None:
                            if k in close and not accepted:portals.append(dict(record,verdict=miss))
                            continue
                        for s,base in ((0.,lane.axis[j]),(length,lane.axis[j+1])):
                            if (s<=EPS and (j!=0 or windows[j][0]>EPS)) or (s>EPS and (j!=len(chart)-1 or windows[j][1]<1.-EPS)):continue
                            up=railride.support_normal(t); side=sc.unit(np.cross(up,t)); rad=r+lane.tube
                            h0=float(base@p.normal-p.offset-r)
                            for angle in rimclear.trig_zeros(h0,rad*(up@p.normal),rad*(side@p.normal),-math.pi/2,math.pi/2):
                                n=up*math.cos(angle)+side*math.sin(angle); at=base+rad*n
                                if not p.polygon.covers(Point((at-r*p.normal)[XZ])):continue
                                x=_road_leg(tf,obs,p,at,at,r,t)
                                if x is None:continue
                                l,u=(-length,0.) if s>EPS else (0.,length)
                                curve=crown_arc(tf,obs,ball,base,t,lane.tube,n,up,l-TOL,u+TOL)
                                if curve is None:stats['road_crown_arc_unproved']+=1;miss='crown_arc_bearing_unproved';continue
                                # The same locus is certified in reverse;
                                # no speed or input schedule was used in it.
                                _,turn,*_=curve.arc
                                rv=curve.v*math.cos(turn)+curve.b*math.sin(turn)
                                rb=curve.v*math.sin(turn)-curve.b*math.cos(turn)
                                reverse=motion.arc(base,rv,rb,0.,turn,np.zeros(3),(1.,0.,0.))
                                end=curve.point(curve.end); rep=(a+b)*.5
                                if np.linalg.norm(end-(a+s*t))>EPS:continue
                                link(k,nodes[j],ball,'road_to_crown_arc',((x,at),curve,(end,rep)))
                                link(nodes[j],k,ball,'crown_arc_to_road',((rep,end),reverse,(at,x)))
                                accepted=True;portals.append(dict(record,verdict='supported_crown_arc'))
                        if k in close and not accepted:portals.append(dict(record,verdict=miss))
    # Separate conformal chains may terminate at the same contact locus.
    # Never infer a handoff merely because two rails share an owner or XZ.
    from scipy.spatial import cKDTree
    for ball in ph.BALLS.values():
        table=ends[ball.name]
        if not table:continue
        for j,k in sorted(cKDTree([e[3] for e in table]).query_pairs(EPS)):
            i,u,t,p,a,typ=table[j]; h,v,q,x,b,other=table[k]
            if i==h or u==v or (typ=='groove')!=(other=='groove') or t@q>=-EPS:continue
            link(u,v,ball,'coincident_chain_ends',((a,p),(x,b)))
            link(v,u,ball,'coincident_chain_ends',((b,x),(p,a)))
    unique={}
    for q in portals:
        key=tuple(q[k] for k in ('chain','line','piece','patch','ball'))
        if key not in unique or not unique[key]['verdict'].startswith('supported'):
            unique[key]=q
    portals=list(unique.values())
    g.info['quasistatic_rail']=dict(stats=dict(stats),pieces=rows,joints=joints,edges=edges,exits=exits,
                                   road_portals=portals,road_portal_reasons=dict(Counter(q['verdict'] for q in portals)))
