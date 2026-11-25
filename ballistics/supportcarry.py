"""Carry momentum through supported pieces; all motion is evaluated by events."""
from dataclasses import dataclass
import math

import numpy as np

from . import closedform as cf, physics as ph, motion, rails, terrain, conformal
from .coarse import STEP, WALK_NY, _lane_chains
from .geometry import _closest

EPS = 1.e-7
UP = np.array([0., 1., 0.])
XZ = [0, 2]
MAX_CARRIES = 8  # ASSUMED: search budget for new flights, not an engine limit
MAX_SUPPORT_PIECES = 256  # ASSUMED: bounds patch traversal, not flight time
SKIN = ph.Params().floor
RECAPTURE = ph.Params().touch  # MEASURED contact band; using it as apex cutoff is ASSUMED


def unit(v):
    size = np.linalg.norm(v)
    return v / size if size > EPS else np.zeros_like(v)


def inertia_ratio(ball):
    # MEASURED: balls/README sections 2 and 6; effective paper approximation.
    return cf.inertia_ratio(ball)


def law(ball, normal, push):
    k = inertia_ratio(ball)
    b = cf.air_impulse(ball, push=push)
    return cf.RollingLaw((b - (b @ normal) * normal) / (1 + k),
                         ph.DT * (ball.lin_damp + k * ball.rot_damp) / (1 + k))


def rolling_entry(ball, velocity, normal, omega=None):
    """Conserve tangential linear + angular impulse through slip capture."""
    tangent = velocity - (velocity @ normal) * normal
    k = inertia_ratio(ball)
    spin = np.zeros(3) if omega is None else ball.radius * np.cross(omega, normal)
    return (tangent + k * spin) / (1 + k)


def keys_along(v):
    # Nearest of the eight world input directions by angle; diagonal push
    # still has two full impulse components. A numerical yaw is not a key.
    x, z = v[XZ]
    if max(abs(x),abs(z)) <= EPS:
        return np.zeros(2)
    if abs(z) <= (math.sqrt(2)-1)*abs(x):
        return np.array([math.copysign(1.,x),0.])
    if abs(x) <= (math.sqrt(2)-1)*abs(z):
        return np.array([0.,math.copysign(1.,z)])
    return np.array([math.copysign(1.,x),math.copysign(1.,z)])


def local_speed(axis):
    """Speed a ball from rest earns in one coarse spacing of this support."""
    end = axis.over(0., STEP)
    return max(math.sqrt(2 * ph.GRAVITY * SKIN), end[0] if end else 0.)


def has_carry_speed(ball, normal, roll):
    speed=np.linalg.norm(roll)
    return speed>EPS and speed>local_speed(law(ball,normal,keys_along(roll)).along(roll))+EPS


def slow_tick(axis, speed):
    threshold = local_speed(axis)
    if speed <= threshold + EPS:
        return 0.
    terminal = axis.terminal_speed
    if terminal >= threshold:
        return math.inf
    return math.log((threshold - terminal) / (speed - terminal)) / math.log1p(-axis.loss)


def apex(ball, v, n, push):
    vn = float(v @ n)
    if vn <= EPS:
        return 0.
    b = float(cf.air_impulse(ball, push=push) @ n)
    if b >= 0.:
        return math.inf
    value, _ = cf._motion(cf._turn(vn, b, ball.lin_damp * ph.DT), vn, b, ball.lin_damp * ph.DT)
    return max(0., float(value))


def lateral_exit(ball, gauge, tube, transverse_speed, up):
    """Ideal symmetric double rail: energy needed to crest the outer tube."""
    r = ball.radius + tube + SKIN
    half = gauge / 2.
    if not EPS < half < r or up[1] <= 0:
        return None
    rise = r - math.sqrt(r*r - half*half)
    left = transverse_speed**2 - 2 * ph.GRAVITY * up[1] * rise / (1 + inertia_ratio(ball))
    if left <= 0.:
        return None
    # Spatial model for the traverse: arc about the outside tube, conservative
    # duration bound from terminal transverse speed (ASSUMED representative).
    distance = r * math.asin(half / r)
    speed = math.sqrt(left)
    ticks = 2 * distance / (abs(transverse_speed) + speed) / ph.DT
    return half, rise, speed, ticks


def segment(kind, obj, a, b, ticks, push, reason='', *, geometry=None):
    return dict(kind=kind, object=obj, start_position=a.pos.copy(), end_position=b.pos.copy(),
                start_velocity=a.vel.copy(), end_velocity=b.vel.copy(),
                ticks=float(ticks), time=float(ticks / 132.), push=np.asarray(push).copy(), reason=reason,
                motion=(motion.line(a.pos,b.pos),) if geometry is None else geometry)


@dataclass
class Support:
    reason: str
    body: ph.State
    parts: list
    way: list
    node: int = -1
    rim: object = None
    departure: object = None
    push: object = None


class SurfacePaths:
    """Geometry index, shared across the surviving proposals of one graph."""
    def __init__(self, g, obs, landing):
        self.g, self.obs, self.landing = g, obs, landing
        self.lanes = {int(g.lane[ids[0]]): ids for ids in _lane_chains(g) if len(ids)}
        self.rail_geometry = rails.physical_lanes(g,obs.mesh)
        self.floor_edges = None
        self.rail_bends = None

    def name(self, face):
        return self.obs.mesh.entities[self.obs.mesh.owner[face]].get('name', '')

    def floor_neighbours(self):
        if self.floor_edges is None:
            self.floor_edges = edges = {}
            m = self.obs.mesh
            for i, tri in enumerate(self.obs.tris):
                if m.rail[i] or m.placed[i] or abs(m.N[i, 1]) < WALK_NY:
                    continue
                for j in range(3):
                    key = self.edge_key(tri[j], tri[(j+1) % 3])
                    edges.setdefault(key, []).append(i)
        return self.floor_edges

    @staticmethod
    def edge_key(a, b):
        return tuple(sorted((tuple(np.round(a, 5)), tuple(np.round(b, 5)))))

    def rail_curvature(self, point, before, after, reach):
        """Curvature of a matched bend in the mesh-derived tube axes.

        Three *mesh ring centres* define the local osculating circle on a
        polygonal bend. A straight run or missing match supplies no radius.
        """
        if self.rail_bends is None:
            from .rails import axis_bends
            self.rail_bends = axis_bends(self.obs.mesh.tubes)
        tree, bends = self.rail_bends
        if tree is None:
            return 0.
        choices = []
        for i in tree.query_ball_point(point,reach):
            p,u,v,k = bends[i]
            if (-v)@before + (-u)@after > u@before + v@after:
                u,v = -v,-u
            if min(u@before,v@after)<.8:  # ASSUMED matching tolerance, not a curvature estimate
                continue
            choices.append((np.linalg.norm(p-point), k))
        return min(choices)[1] if choices else 0.

    def clear(self, tf, ball, points, owner, normal):
        # Rails are straight geometric segments between analytic events. The
        # time reparametrization is immaterial for static geometry.
        for a,b in zip(points,points[1:]):
            vel = (b-a)/ph.DT
            if not self.obs.support_clear(ph.State(a,vel), np.zeros(3), .01, 1., owner, ball.radius-RECAPTURE):
                return False
        return True

    def impact(self, tf, ball, hit, node, omega=None, chosen_push=None, controls=None, clock=0.):
        face = hit['triangle']
        omega=hit.get('incoming_omega',omega)
        start = ph.State(hit['position'], hit['incoming'], np.zeros(3) if omega is None else omega)
        normal = hit['normal']
        if hit['kind'] == 'dual_rail' and 'support_normal' in hit:
            normal = hit['support_normal']
        material = self.obs.mesh.entities[self.obs.mesh.owner[face]]
        reflected, _ = cf.collision_impulse(ball, start, normal, float(material.get('elasticity', tf.DEFAULT_ELASTICITY)))
        roll = rolling_entry(ball, reflected.vel, normal, reflected.omega)
        push = keys_along(roll) if chosen_push is None else chosen_push
        # Only carry momentum exceeding what the same local support can
        # supply from rest. The rail longitudinal gate is applied below.
        if not has_carry_speed(ball,normal,roll):
            return Support('handoff', reflected, [], [], node)
        if apex(ball, reflected.vel, normal, push) > RECAPTURE:
            if 'point' in hit and hit['kind'] != 'dual_rail':
                reflected.pos = hit['point']+(ball.radius+SKIN+8*EPS)*normal
            else:
                reflected.pos = reflected.pos + 8 * EPS * normal
            if self.obs.mesh.rail[face] and node >= 0 and int(self.g.lane[node]) in self.lanes:
                pts = self.rail_geometry[int(self.g.lane[node])]
                if len(pts)>1:
                    j,_=rails.project_pieces(pts,start.pos)
                    axis = pts[j+1]-pts[j]
                    push = keys_along(axis if axis @ roll >= 0 else -axis)
            return Support('rebound', reflected, [], [], push=push)
        body = ph.State(start.pos, roll, ph.rolling_spin(roll, normal, ball.radius))
        if self.obs.mesh.rail[face]:
            return self.rail(tf, ball, face, node, body, chosen_push, controls, clock)
        return self.floor(tf, ball, face, node, body, normal, chosen_push, controls, clock)

    def rail(self, tf, ball, face, node, body, chosen_push=None, controls=None, clock=0.):
        g, obs = self.g, self.obs
        lane = int(g.lane[node]) if node >= 0 else -1
        if lane not in self.lanes:
            return Support('no_lane', body, [], [])
        ids = self.lanes[lane]; pts = self.rail_geometry[lane]; meta = g.lanes[lane]
        if len(pts) < 2:
            return Support('no_lane', body, [], [])
        D = np.diff(pts, axis=0)
        j,center=rails.project_pieces(pts,body.pos)
        tangent=unit(D[j])
        forward = 1 if tangent @ body.vel >= 0 else -1
        tangent = forward * tangent
        normal = unit(UP - (UP @ tangent) * tangent)
        speed = body.vel @ tangent
        push = keys_along(tangent) if chosen_push is None else chosen_push
        axis = law(ball, normal, push).along(tangent)
        if speed <= local_speed(axis) + EPS:
            return Support('handoff', body, [], [], node)
        way, pieces = [body.pos], []
        obj = self.name(face)
        # Lateral ejection only from the modelled double-rail valley. The
        # entry projection is local; do not teleport from outside the tube.
        side = unit(np.cross(normal,tangent))
        cross = body.vel @ side
        lateral = lateral_exit(ball, meta['gauge'], meta['radius'], cross, normal) if meta['double'] else None
        if lateral and (controls is None or lateral[-1] <= controls(clock)[0][0]):
            half, rise, vs, ticks = lateral
            s = axis.at(speed,ticks)
            end = center + math.copysign(half,cross)*side + rise*normal + float(s[0])*tangent
            left = (pts[j+(forward>0)]-center) @ tangent
            if float(s[0]) <= left + EPS:
                at = ph.State(end, float(s[1])*tangent + math.copysign(vs,cross)*side)
                if not self.clear(tf,ball,[body.pos,end],obs.mesh.owner[face],normal):
                    return Support('support_obstruction',body,[],[])
                return Support('side_release',at,[segment('rail',obj,body,at,ticks,push,'side_energy')],[body.pos,end],push=push)
        entry = ph.State(center, speed*tangent, ph.rolling_spin(speed*tangent,normal,ball.radius))
        way.append(center)
        # Every vertex is a joint of mesh-derived straight pieces.
        elapsed = 0.
        for _ in range(MAX_SUPPORT_PIECES):
            limit = math.inf
            if controls is not None:
                limit,push = controls(clock+elapsed)[0]
                if limit <= EPS:
                    return Support('no_input_window',entry,pieces,way)
                axis = law(ball,normal,push).along(tangent)
            k = j+1 if forward>0 else j
            end = pts[k]
            nxt = k if forward>0 else k-1
            distance = max(0., float((end-entry.pos) @ tangent))
            solved = axis.over(speed,distance)
            tslow = slow_tick(axis,speed)
            handoff = solved is None or tslow < solved[1] / ph.DT
            if handoff:
                if not math.isfinite(tslow):
                    return Support('no_forward_root', entry,pieces,way)
                dx,next_speed = axis.at(speed,tslow)
                end = entry.pos + float(dx)*tangent
                ticks = tslow
            else:
                next_speed,seconds = solved; ticks = seconds / ph.DT
            switch = limit < ticks - EPS
            if switch:
                dx,next_speed = axis.at(speed,limit)
                end = entry.pos + float(dx)*tangent
                ticks = limit
            elapsed += ticks
            out = ph.State(end,float(next_speed)*tangent,ph.rolling_spin(float(next_speed)*tangent,normal,ball.radius))
            if not self.clear(tf,ball,[entry.pos,end],obs.mesh.owner[face],normal):
                return Support('support_obstruction',entry,pieces,way)
            if ticks > EPS:
                pieces.append(segment('rail',obj,entry,out,ticks,push,'key_change' if switch else 'slow' if handoff else 'chain_piece'))
                way.append(end)
            if switch:
                entry,speed = out,float(next_speed)
                continue
            if handoff:
                q = int(ids[np.argmin(np.linalg.norm(g.pos[ids]-end,axis=1))])
                return Support('handoff',out,pieces,way,q)
            if not 0 <= nxt < len(D):
                cap = tf.rail_lip(obs,end,normal,tangent,meta['double'],ball.radius)
                rim = None
                if cap is not None:
                    pivot,n0,forward0,radius = cap
                    rim = tf.rim_release(ball,pivot,n0,forward0,out.vel,0,radius, inertia_ratio=inertia_ratio(ball), push=push)
                if rim is not None and rim.seconds > 0:
                    pieces.append(segment('rim',obj,out,rim.start,rim.seconds/ph.DT,push,'rail_end',geometry=(rim.motion(),)))
                    way.extend(rim.way())
                    out = rim.start
                departure = tf.rail_support(obs,end,ball.radius)
                return Support('chain_end',out,pieces,way,rim=rim,departure=departure,push=push)
            new_t = forward*unit(D[nxt]); new_n = unit(UP - (UP @ new_t) * new_t)
            # Centreline convex kink: two offset normals define the virtual
            # rim. The energy/pressure test chooses the tangential lift-off.
            curve = self.rail_curvature(end,tangent,new_t,ball.radius+meta['radius']+SKIN)
            pressing = float((cf.air_impulse(ball,push=push)/ph.DT) @ normal) + next_speed**2*curve
            if new_t @ normal < -EPS and pressing >= 0.:
                rim = tf.convex_release(ball,end,normal,new_n,tangent,out.vel,push,inertia_ratio(ball))
                if rim:
                    pieces.append(segment('rim',obj,out,rim.start,rim.seconds/ph.DT,push,'convex',geometry=(rim.motion(),)))
                    way.extend(rim.way())
                    return Support('convex',rim.start,pieces,way,rim=rim)
            # A concave bend adds normal force. Track direction change by
            # inelastic tangent projection (not an upward-turn release).
            speed = float(out.vel @ new_t)
            if speed <= EPS:
                return Support('handoff',out,pieces,way,int(ids[np.argmin(np.linalg.norm(g.pos[ids]-out.pos,axis=1))]))
            j,tangent,normal = nxt,new_t,new_n
            push = keys_along(tangent) if chosen_push is None else chosen_push
            axis = law(ball,normal,push).along(tangent)
            entry = ph.State(end,speed*tangent,ph.rolling_spin(speed*tangent,normal,ball.radius))
        return Support('support_budget',entry,pieces,way)

    def climb_lip(self, tf, ball, lip, pivot, body, push, remaining):
        obs=self.obs; m=obs.mesh
        r=ball.radius+SKIN
        forward=np.array([lip.forward[0],0.,lip.forward[1]])
        height=r-(body.pos-pivot)[1]
        if not 0.<height<ball.radius:return None
        inertia=inertia_ratio(ball)
        speed=float(body.vel@forward)
        result=terrain.step_energy(ball,height,speed,lip.forward,inertia,push=push,radius=r)
        if result is None or result[0]<=EPS:return None
        _,crown_speed=result
        theta=math.acos(1-height/r)
        force=terrain.ratio(ball)*ph.GRAVITY*float(push@lip.forward)
        pivot_speed=max(0.,speed)*(1-height/(r*(1+inertia)))
        energy=(pivot_speed**2,-2*r*ph.GRAVITY/(1+inertia),-2*r*force/(1+inertia))
        path=motion.arc(pivot,r*UP,-r*forward,theta,-theta,np.zeros(3),energy)
        ticks=path.duration(1.)/ph.DT
        if not math.isfinite(ticks) or ticks>remaining:return None
        lo,hi=path.bounds(0.,1.)
        ids=tf._box_triangles((lo-r)[None],(hi+r)[None],obs.lo,obs.hi,m._grid)
        # Check every unrelated obstacle against the continuous arc. Only
        # the three bearing pieces of this two-plane lip are exempted below.
        surface=self.g.info['conformal']
        idx=[int(i) for i in ids if not (surface.face_patch[i] in (lip.lower,lip.upper) or
              int(m.owner[i])==lip.owner and abs(m.N[i,1])<=math.sqrt(1-conformal.ANGLE**2) and
              abs((obs.tris[i,0]-pivot)@forward)<=conformal.PLANE_ERROR)]
        if idx:
            selector=np.zeros(len(m.A),bool);selector[idx]=True
            nearby=m.subset(selector)
            if motion.near_hit(path,lambda q:nearby.distance(q,r+1.)[0],ball.radius-RECAPTURE) is not None:
                return None
        if any(motion.box_hit(path,a,b) is not None for a,b in obs.boxes):return None
        result=ph.State(path.point(1.),crown_speed*forward,ph.rolling_spin(crown_speed*forward,UP,ball.radius))
        material=m.entities[lip.owner] if lip.owner>=0 else {}
        limits=terrain.step_contact_limits(height,r,force,inertia,pivot_speed,ball.friction*float(material.get('friction',.7)))
        # An ideal-roll witness is not proof under the original contact
        # solver when it requires tension/slip or an oblique impact.
        aligned=np.linalg.norm(body.vel-speed*forward)<=EPS and abs(push@np.array([-forward[2],forward[0]]))<=EPS
        # Keep the proposal for fine validation, rather than interpreting
        # the omitted slip/detachment physics as proof of a blocked route.
        verdict='ideal_step_roll' if aligned and min(limits)>=0. else 'step_contact_unproved'
        return result,ticks,path,verdict,limits

    def floor(self, tf, ball, face, node, body, normal, chosen_push=None, controls=None, clock=0.):
        obs,m = self.obs,self.obs.mesh
        parts,way = [],[body.pos]
        elapsed = 0.
        returning_push = None
        surface = getattr(self.g,'info',{}).get('conformal')
        for _ in range(MAX_SUPPORT_PIECES):
            tri = obs.tris[face]
            k_patch = surface.face_patch[face] if surface is not None else -1
            patch = surface.patches[k_patch] if k_patch>=0 else None
            plane = patch.normal if patch is not None else m.N[face] * np.sign(m.N[face] @ normal)
            offset = patch.offset if patch is not None else float(plane @ tri[0])
            if plane[1] < WALK_NY:
                return Support('handoff',body,parts,way,node)
            vel = body.vel - (body.vel @ plane)*plane
            speed = np.linalg.norm(vel)
            if speed <= EPS and returning_push is None:
                return Support('handoff',body,parts,way,node)
            push = returning_push if returning_push is not None else (keys_along(vel) if chosen_push is None else chosen_push)
            limit = math.inf
            if controls is not None:
                limit,push = controls(clock+elapsed)[0]
                if limit <= EPS:
                    return Support('no_input_window',body,parts,way)
            rolling = law(ball,plane,push)
            stop = slow_tick(rolling.along(vel),speed)
            stalled=False
            if patch is not None and self.g.info.get('terrain_ball_masks') is not None:
                grade=np.linalg.norm(plane[XZ])/plane[1]
                if terrain.slope_verdict(ball,grade,plane[XZ])<=0:
                    up=unit(UP-plane[1]*plane)
                    turn=terrain.climb_turn(rolling,up,vel)
                    if math.isfinite(turn):
                        stop,stalled=turn,True
                    elif returning_push is not None:
                        stop=math.inf
            ground = body.pos - (body.pos @ plane-offset)*plane
            start = ph.State(ground+(ball.radius+SKIN)*plane,vel,body.omega)
            boundary = patch.boundary if patch is not None else tuple(
                (face,a,b) if m.N[face]@plane>=0 else (face,b,a) for a,b in zip(tri,np.roll(tri,-1,axis=0)))
            first = math.inf
            crossing = None
            if patch is not None:
                from shapely.geometry import Point
                outside = not patch.polygon.covers(Point(ground[XZ]))
            else:
                outside = False
            rim_distance = math.inf
            for f,a,b in boundary:
                edge=b-a
                edge=edge-(edge@plane)*plane
                n=unit(np.cross(edge,plane))
                gap=float((a-ground)@n)
                u,j=float(vel@n),float(rolling.impulse@n)
                if u<=EPS and j<=EPS:continue
                if outside and gap < -EPS and u > EPS:
                    # A radial hit can support a ball whose orthogonal floor
                    # projection is already outside. An outward crossing of
                    # the *touching finite edge* happened at entry, not at a
                    # future zero of the signed plane distance.
                    edge3=b-a
                    f0=float(np.clip((body.pos-a)@edge3/(edge3@edge3),0.,1.))
                    distance=np.linalg.norm(body.pos-a-f0*edge3)
                    if distance <= ball.radius+SKIN+EPS and distance<rim_distance:
                        first=0.; crossing=(f,a,b,n); rim_distance=distance
                    continue
                lo=cf._turn(u,j,rolling.loss) if u<0<j else 0.
                hi=cf._turn(u,j,rolling.loss) if j<0 else math.inf
                hi=min(hi,first,stop,limit)
                if lo>hi:continue
                root=cf._increasing_root(u,j,rolling.loss,gap,lo,hi)
                if root is None:continue
                displacement,v=rolling.at(vel,root)
                if abs(displacement@n-gap)>EPS:continue
                along=float((ground+displacement-a)@edge)
                if not -EPS<=along<=edge@edge+EPS:continue
                if v@n<=EPS and j<=EPS:continue
                if root<first:
                    first=root; crossing=(f,a,b,n)
            t=first
            lips=self.g.info.get('terrain_lips',{}).get(k_patch,()) if patch is not None else ()
            lip_event=terrain.meet_step(surface,lips,ball,start,rolling,min(t,stop,limit)) if lips else None
            if lip_event is not None:t=lip_event[0]
            if not math.isfinite(min(t,stop,limit)):
                return Support('support_unbounded',body,parts,way)
            event = min(t,stop,limit)
            dx,v = rolling.at(vel,event)
            out = ph.State(start.pos+dx,v,ph.rolling_spin(v,plane,ball.radius))
            if not self.obs.support_clear(start,rolling.impulse,rolling.loss,event,m.owner[face],ball.radius-RECAPTURE):
                return Support('support_obstruction',body,parts,way)
            # Only the exported path uses a display polyline.
            _, points = tf._samples(start,rolling.impulse,rolling.loss,event,1.)
            parts.append(segment('floor',self.name(face),start,out,event,push,'key_change' if limit<min(t,stop) else ('uphill_turn' if stalled else 'slow') if stop<=t else 'step_contact' if lip_event is not None else 'patch_edge' if patch is not None else 'face_edge',
                                 geometry=(motion.rolling(start,rolling.impulse,rolling.loss,event),)))
            way.extend(points)
            elapsed += event
            if limit < min(t,stop):
                body = out
                continue
            if stop <= t:
                if stalled:
                    returning_push=push
                    # Keep the recursive velocity state: at the stationary
                    # point of position it is slightly negative (PSI phase).
                    body=out
                    continue
                v = self.landing.near(face,out.pos)
                return Support('handoff',out,parts,way,v)
            if lip_event is not None:
                _,lip,pivot=lip_event
                result=self.climb_lip(tf,ball,lip,pivot,out,push,limit-event)
                if result is None:
                    return Support('step_energy_or_obstacle',out,parts,way)
                body,ticks,path,verdict,limits=result
                parts.append(dict(segment('step',self.name(lip.face),out,body,ticks,push,verdict,geometry=(path,)),
                                  contact_margins=limits))
                way.append(body.pos); elapsed+=ticks
                face=lip.face; normal=m.N[face]; returning_push=None
                continue
            f,a,b,outward=crossing
            adjacent = self.floor_neighbours().get(self.edge_key(a,b),())
            candidates = [i for i in adjacent if i != f and (patch is None or surface.face_patch[i]!=k_patch)]
            if candidates:
                other = max(candidates,key=lambda i: abs(m.N[i] @ plane))
                n1 = m.N[other] * np.sign(m.N[other] @ plane)
                if out.vel @ n1 > EPS:
                    rim = tf.convex_release(ball,out.pos,plane,n1,outward,out.vel,push,inertia_ratio(ball))
                    if rim:
                        parts.append(segment('rim',self.name(face),out,rim.start,rim.seconds/ph.DT,push,'convex',geometry=(rim.motion(),)))
                        way.extend(rim.way()); return Support('convex',rim.start,parts,way,rim=rim)
                # Re-anchor to the same edge's offset on the neighbouring face.
                p = out.pos - (ball.radius+SKIN)*plane + (ball.radius+SKIN)*n1
                body = ph.State(p,out.vel,out.omega); face,normal = other,n1
                returning_push=None
                continue
            pivot = out.pos-(ball.radius+SKIN)*plane
            rim = tf.rim_release(ball,pivot,plane,outward,out.vel,0,inertia_ratio=inertia_ratio(ball),push=push)
            if rim is not None and rim.seconds > 0:
                parts.append(segment('rim',self.name(face),out,rim.start,rim.seconds/ph.DT,push,'floor_end',geometry=(rim.motion(),)))
                way.extend(rim.way()); out = rim.start
            return Support('floor_end',out,parts,way,rim=rim)
        return Support('support_budget',body,parts,way)
