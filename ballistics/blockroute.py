"""Conformal geometry, graph identities, and the temporary mechanism bridge."""
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from shapely import MultiPoint, Point, STRtree, union_all
from shapely.ops import nearest_points

from . import coarse, conformal, motion, physics as ph, rails, supportcarry as sc
from .conformalsearch import Rail

XZ = [0, 2]


def _clip(poly, normal, value):
    """Clip the triangle against a 3-D halfspace by affine intersections."""
    out = []
    for a, b in zip(poly, (*poly[1:], poly[0])):
        da, db = normal @ a - value, normal @ b - value
        if da >= 0.:
            out.append(a)
        if (da < 0.) != (db < 0.):
            out.append(a + da / (da - db) * (b - a))
    return out


def stand_ranges(surface, mesh, layout):
    """Conservative polygonal inner ranges, per supported patch and ball.

    Centre projection is the plane's footprint translated by its normal.
    Features have to intersect the sphere's support slab to obstruct it.
    A circumscribed square bounds the horizontal disk; subtract its convex
    sum with each clipped triangle. No centre lattice is queried.
    """
    from shapely import box
    from shapely.affinity import translate
    tri = surface.triangles
    if not len(tri):
        return {b.name: {} for b in ph.BALLS.values()}
    lo, hi = tri.min(1), tri.max(1)
    tree = STRtree([box(*np.r_[a[XZ], b[XZ]]) for a, b in zip(lo, hi)])
    out = {}
    for ball in ph.BALLS.values():
        r = ball.radius + sc.SKIN
        corners = r * np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        ranges = out[ball.name] = {}
        for patch_id, patch in enumerate(surface.patches):
            if not conformal_search_rest(ball, patch):
                continue
            footprint = translate(patch.polygon, *(r * patch.normal[XZ]))
            lo2, hi2 = np.asarray(footprint.bounds).reshape(2, 2)
            obstructions = []
            for i in tree.query(box(*np.r_[lo2-r, hi2+r])):
                if surface.face_patch[i] == patch_id or mesh.placed[i]:
                    continue
                projections = tri[i] @ patch.normal
                lower = patch.offset + conformal.PLANE_ERROR
                upper = patch.offset + 2*r
                if projections.max() <= lower or projections.min() >= upper:
                    continue
                polygon = _clip(list(tri[i]), patch.normal, lower)
                if not polygon:
                    continue
                polygon = _clip(polygon, -patch.normal, -upper)
                if polygon:
                    shadow = MultiPoint((np.asarray(polygon)[:, None, XZ]+corners).reshape(-1, 2)).convex_hull
                    if footprint.intersects(shadow):
                        obstructions.append(shadow)
            ranges[patch_id] = footprint.difference(union_all(obstructions)) if obstructions else footprint
    return out


def conformal_search_rest(ball, patch):
    from .conformalsearch import _rest
    return _rest(ball, patch)


@dataclass
class GeometryGraph(coarse.Graph):
    """Patch and finite rail identity adapter, not a standing-position grid."""

    def nearest(self, p, reach=6., floor_only=True):
        patch_ids = self.patch_ids(p, reach)
        if floor_only:
            return patch_ids[0] if patch_ids else -1
        u = self.rail_at(p, reach)
        if u is not None:
            return u
        return patch_ids[0] if patch_ids else -1

    def patch_ids(self, p, reach=6.):
        x = np.asarray(p, float)
        surface = self.info['conformal']
        if not surface.patches:
            return ()
        from shapely import box
        from shapely.affinity import translate
        index = getattr(self, '_patch_index', None)
        r = ph.WOOD.radius+sc.SKIN
        if index is None:
            polygons = [translate(f.polygon, *(r*f.normal[XZ])) for f in surface.patches]
            index = self._patch_index = (polygons, STRtree(polygons))
        q = Point(x[XZ]); candidates = []
        for j in index[1].query(box(*(np.r_[x[XZ]-reach,x[XZ]+reach]))):
            foot = np.array(nearest_points(q,index[0][j])[1].coords[0]); f = surface.patches[j]
            d = np.linalg.norm(x-np.array([foot[0],f.height(foot,r),foot[1]]))
            if d <= reach:
                candidates.append((float(d),int(j)))
        if not candidates:
            return ()
        candidates.sort()
        return tuple(j for d,j in candidates if d <= candidates[0][0] + conformal.PLANE_ERROR)

    def rail_at(self, p, reach, *, chain=None):
        best = (reach, None)
        for i, lane in enumerate(self.lanes):
            if chain is not None and i != chain:
                continue
            pts = lane['segments']
            if len(pts)<2:
                continue
            j,q = rails.project_pieces(pts, p)
            d = np.linalg.norm(q-p)
            if d < best[0]:
                best = d,self.info['rail_piece_ids'][i][j]
        return best[1]


def geometry_graph(surface, metadata, lips):
    p = []
    for face in surface.patches:
        q = np.array(face.polygon.representative_point().coords[0])+ (ph.WOOD.radius+sc.SKIN)*face.normal[XZ]
        p.append([q[0],face.height(q,ph.WOOD.radius+sc.SKIN),q[1]])
    n = len(p); lane_idx = [-1]*n; ids = []
    for i, rail in enumerate(metadata):
        seq=rail['segments']
        ids.append(tuple(range(len(p),len(p)+len(seq))))
        p.extend(seq);lane_idx.extend([i]*len(seq))
    return GeometryGraph(np.asarray(p).reshape(-1,3), np.array(lane_idx)>=0, np.asarray(lane_idx,dtype=int),
                         [[] for _ in p],metadata,dict(native_geometry=True,conformal=surface,terrain_lips=lips,
                              rail_piece_ids=tuple(ids),rest_places=n,edge_node=np.ones(len(p),bool),
                              terrain_support=tuple((k,) for k in range(n)),launches={}))


def _publish(g, u, v, ball, row, status, kind, reference=None):
    from . import takeoff
    if u is None or v is None or u==v or min(u,v)<0:
        return
    origin,end=np.asarray(row['origin']),np.asarray(row['position'])
    start=np.asarray(row.get('launch',origin))
    velocity=np.asarray(row.get('velocity',row.get('origin_velocity',np.zeros(3))))
    ticks=row.get('ticks',np.linalg.norm(end-origin)/(takeoff.FOLLOW_SPEED*ph.DT))
    e=dict(ball=ball, ticks=ticks, flight_ticks=row.get('flight_ticks',0.), verified=not status.startswith('retained'),
           witness_status=status,route_type=kind,retained_reference=reference,
           support_patches=(row['source_patch'],row['target_patch']), launch_position=start,
           launch_velocity=velocity,landing_position=end,first_landing_node=v,
           landing_face=row.get('face',-1),launch_speed=float(np.linalg.norm(velocity)),
           motion=row['motion'],contacts=row.get('contacts',()),rail_parts=row.get('rail_parts',()),
           key_table=row.get('key_table',()),shared_supported_transfer=True,
           way=np.vstack((row['origin'],*[p.point(p.end) for p in row['motion']],end)),
           across=np.linalg.norm((start-end)[XZ]),down=start[1]-end[1])
    prior=takeoff.variants(g,u,v)
    variants=tuple(sorted((*prior,e),key=lambda x:(not x['verified'],x['ticks'])))
    g.info['launches'][u,v]=dict(variants[0],variants=variants)
    if not prior:
        g.adj[u].append((v,takeoff.cost(e),'launch'))
    elif variants[0] is e:
        g.adj[u]=[(j,takeoff.cost(e) if j==v and k=='launch' else c,k) for j,c,k in g.adj[u]]


def attach(g, layout, *, obs=None):
    shared=g.info['shared_motion_graph']; masks={}; proof=defaultdict(list)
    def add(u,v,ball,rule,span,kind='roll',motion=(),reverse=False):
        if u==v:
            return
        k=(u,v,kind)
        if k not in masks:
            g.adj[u].append((v,max(sc.EPS,float(np.linalg.norm(g.pos[v]-g.pos[u]))),kind))
        masks[k]=masks.get(k,0)|(1<<coarse.KIND_INDEX[ball]);proof[u,v].append(dict(ball=ball,rule=rule,span=span,motion=motion,reverse=reverse))
    for ball, labels in layout.verdicts.items():
        for seam, verdict in zip(layout.seams, labels):
            if verdict=='smooth':
                a,b=seam.patches
                add(a,b,ball,'smooth_seam',seam.endpoints);add(b,a,ball,'smooth_seam',seam.endpoints)
    for rec in g.info['portal_verdicts']:
        if rec['status']!='verified':
            continue
        row=rec.get('witness',{})
        if rec['patch_pair'] is not None and row.get('kind')=='roll':
            a,b=rec['patch_pair'];add(a,b,rec['source'].ball,rec['reason'],rec['segment'],motion=row.get('motion',()))
        elif rec['kind']=='rail_ring':
            a,b=rec['source'],rec['target']
            if a.line not in ('groove','single'):
                continue
            p,q=(layout.rail_segments[x].mean(0) for x in (a,b))
            u,v=g.rail_at(p,3.,chain=a.chain),g.rail_at(q,3.,chain=b.chain)
            if u is not None and v is not None:
                add(u,v,a.ball,rec['reason'],rec['segment'],kind='rail')
    if obs is not None:
        from . import blockcontacts, takeoff
        known={(a,b,r['ball']) for (a,b),rows in proof.items() for r in rows}
        certs,limits=blockcontacts.certify(takeoff,obs,g,layout,known)
        g.info.update(block_offset_certificates=dict(certs),block_offset_unresolved=limits)
        for ball,rows in certs.items():
            for (a,b),r in rows.items():
                span=r.get('centers', (r.get('vertex'),)*2)
                add(a,b,ball,r['rule'],span,motion=r['motion']);add(b,a,ball,r['rule'],span)
    if obs is not None:
        from . import railstatic
        railstatic.certify(takeoff,g,obs,layout,add,lambda *args: _publish(g,*args))
    g.info.update(terrain_ball_masks=masks,block_roll_proofs=dict(proof),node_geometry='conformal_patch')
    for (a,b), rows in shared['witnesses'].items():
        for row in rows:
            u=row['source_patch']; v=row['target_patch']
            if u<0 and isinstance(a,Rail):
                u=g.rail_at(row['origin'],3.,chain=a.chain)
            if v<0 and isinstance(b,Rail) and b.line in ('groove','single'):
                v=g.rail_at(row['position'],3.,chain=b.chain)
            kind=('crown_ride' if row.get('crown_ride') else 'shared_rail' if row.get('rail_parts') else 'convex_crest_flight'
                  if row.get('mode','').startswith('convex') else row['mode']
                  if row.get('mode')=='supported_drop' else 'shared_flight')
            _publish(g,u,v,a.ball,row,'crown_ride' if kind=='crown_ride' else 'literal_confirmed',kind)
    # A portal is evidence for its actual landing, not its nominated neighbour.
    seen={id(row) for rows in shared['witnesses'].values() for row in rows}
    for rec in g.info['portal_verdicts']:
        row=rec.get('witness')
        if rec['status']=='verified' and row and row.get('kind')=='air' and id(row) not in seen:
            seen.add(id(row))
            _publish(g,row.get('source_patch'),row.get('target_patch'),rec['source'].ball,row,
                     'literal_confirmed','convex_crest_flight')


def retain_rolling(g, records):
    for (a,_),row in records.items():
        _publish(g,row['source_patch'],row['target_patch'],a.ball,row,'retained_verified_roll',
                 'shared_supported_roll',row.get('retained_reference'))


def archive_for(info):
    from pathlib import Path
    from . import retained
    if 'retained_shared_archive' in info:
        return Path(info['retained_shared_archive'])
    path=retained.archive_for(info)
    return None if path is None else path.with_name('v36-'+path.name.replace('-retained-witnesses','-retained-shared'))


def retain_shared(g, info):
    """Retain signed-geometry witnesses without pretending they were replayed."""
    from . import retained, takeoff
    import json
    file=archive_for(info)
    if file is None or not file.is_file():
        return dict(retained=0,rechecked=0)
    data=json.loads(file.read_bytes())
    if data['map']!=info.get('map') or data['geometry']!=retained.signature(g.info['conformal'].triangles):
        raise ValueError('retained shared archive does not match conformal geometry')
    out=defaultdict(int)
    for archived in data['rows']:
        row=dict(archived['witness']); k=archived['ball']; chain=archived.get('rail_chain')
        u=row['source_patch']; v=row['target_patch']
        if u<0:
            u=g.rail_at(np.asarray(row['origin']),3.,chain=chain)
        if any(x['ball']==k and x['verified'] for x in takeoff.variants(g,u,v)):
            out['rechecked']+=1;continue
        row['motion']=tuple(retained.decode_piece(p) for p in row['motion'])
        _publish(g,u,v,k,row,'retained_verified_shared','shared_rail' if chain is not None else 'shared_flight',data['version'])
        out['retained']+=1
    return dict(out)


def mechanism_state(host, old, base, info, sector, trafos, radius, operate_cost, *, original_host_size, box_mesh=None, box_bodies=()):
    """Translate mechanism endpoints and state guards only, not host rolls.

    ``old`` is the mechanism host's normalized StateGraph. Its state bits
    and transitions are reused, but its ordinary reachability is not.
    """
    from dataclasses import replace
    from . import solve, takeoff
    # Host-normalized float and timed transitions already carry their masks.
    # Virtual lift cabins are mechanism-only nodes, not geometric patches.
    mapping={}; extra=[]
    support=host.info.get('terrain_support',())
    for u in old.glob:
        u=int(u); p=host.pos[u]
        if u>=original_host_size:
            mapping[u]=len(base.pos)+len(extra);extra.append(p);continue
        if host.lane[u]>=0:
            v=base.rail_at(p,3.,chain=int(host.lane[u]))
        elif u<len(support) and support[u]:
            v=min(support[u],key=lambda j:abs(base.info['conformal'].patches[j].height(p[XZ],ph.WOOD.radius+sc.SKIN)-p[1]))
        else:
            v=base.nearest(p,3.)
        if v is not None and v>=0:
            mapping[u]=v
    n=len(extra)
    view=replace(base,pos=np.vstack((base.pos,np.asarray(extra).reshape(-1,3))),kind=np.r_[base.kind,np.zeros(n,dtype=base.kind.dtype)],
                 lane=np.r_[base.lane,np.full(n,-1)],adj=base.adj+[[] for _ in extra],
                 info=dict(base.info,edge_node=np.r_[base.info['edge_node'],np.zeros(n,bool)]))
    rel=coarse.Relaxed(view)
    region=coarse.sector_mask(view.pos,info,sector)
    if mapping:
        region[list(mapping.values())]=True
    tk=np.full(len(view.pos),-1,dtype=int)
    for t in trafos:
        tk[np.linalg.norm(view.pos-t['pos'],axis=1)<radius]=coarse.KIND_INDEX[t['kind']]
    def loc(u):
        return mapping.get(int(old.glob[u]))
    guards={}
    for raw,kind in zip(old._base,old._base_kind):
        u,v,_,_,on,off,_=raw
        a,b=loc(u),loc(v)
        if a is None or b is None or not (on or off) or a==b:
            continue
        key=(a,b,kind); f,c=guards.get(key,(0,0)); guards[key]=(f|on,c|off)
    lifts=[]
    for u,v,c,lk,on,off in old._lift:
        a,b=loc(u),loc(v)
        if a is not None and b is not None:
            lifts.append((a,b,c,lk));guards[a,b,'lift']=(on,off)
    config=[]
    for mc in old.mechs:
        cfg=dict(mc,edges=[])
        if 'rides' in cfg:
            cfg['rides']={(mapping[a],mapping[b],side):row for (a,b,side),row in cfg['rides'].items()
                          if a in mapping and b in mapping}
        if 'deck_edges' in cfg:
            cfg['deck_edges']={(mapping[a],mapping[b],kind):bit for (a,b,kind),bit in cfg['deck_edges'].items() if a in mapping and b in mapping}
        if 'entries' in cfg:
            cfg['entries']=sorted({mapping[u] for u in cfg['entries'] if u in mapping})
        if 'node' in cfg:
            cfg['node']=mapping.get(cfg['node'],-1)
        if 'operate' in cfg:
            cfg['operate']=mapping.get(cfg['operate'],-1)
        config.append(cfg)
    links=solve._link_rules(rel,view,region,tk,trafos,radius,config)
    triggers=[(np.asarray(t['pos'],float),t['kind']) for t in trafos]
    def eligible(u,v,e):
        return takeoff.valid_in_sector(e,e['landing_position'],triggers,radius)
    # Keep gate/board geometry guards on native flight curves, not chords
    # borrowed from host nodes.
    for (u,v),variants in view.info['launches'].items():
        if not region[u] or not region[v]:
            continue
        for j,mc in enumerate(config):
            pieces=mc.get('pieces')
            if mc['type'] not in ('gate','board') or not pieces:
                continue
            hits=[e for e in takeoff.variants(view,u,v) if solve._hits(motion.guard_path_for(e),pieces,0.1)]
            if hits and len(hits)==len(takeoff.variants(view,u,v)):
                guards[u,v,'launch']=(1<<j,0)
    from . import boxmotion
    box_mesh=base.info.get('box_mesh',box_mesh)
    boxes=[]; skip=set(); box_anchors={}
    if box_mesh is not None and box_bodies and np.array_equal(box_mesh.A,base.info['conformal'].triangles[:,0]):
        boxes=boxmotion.state_config(base,box_mesh,box_bodies,info.get('placements',()),sector)
    if boxes:
        radius_ball=max(b.radius for b in ph.BALLS.values())+sc.SKIN
        ids={}
        for item in boxes:
            w=item['push']
            if w is None:continue
            j=w['patch']
            if not region[j] or tk[j]>=0:continue
            cost=max(sc.EPS,w['ticks']*ph.DT*takeoff.FOLLOW_SPEED)
            mask=1<<coarse.KIND_INDEX[w['ball']]; m=len(config)
            ids[item['name']]=m
            item['edges']=[(coarse.BOX_PUSH,j,j,cost,mask,'push_loose_box')]
            config.append(item)
            box_anchors[j,j,m,coarse.BOX_PUSH,mask]=(w['ball_motion'][0].p,w['ball_end'])
        def guard_box(u,v,kind,paths):
            if not paths:return
            for item in boxes:
                m=ids.get(item['name']); q=item['origin']; h=item['half']
                def blocked(c):return any(boxmotion.intersects(path,c,h,radius_ball) for path in paths)
                a=blocked(q); b=a if m is None else blocked(item['push']['box_end'])
                if a and b:
                    skip.add((u,v,kind)); continue
                on,off=guards.get((u,v,kind),(0,0))
                if a:on|=1<<m
                if b:off|=1<<m
                if on or off:guards[u,v,kind]=on,off
        for u in np.flatnonzero(region):
            for v,_,kind in rel.fwd[u]:
                if not region[v] or kind in ('gap','bridge'):continue
                if kind in ('roll','rail'):
                    proofs=base.info.get('block_roll_proofs',{}).get((u,v),())
                    keys=[]
                    for w in proofs:
                        ps=w.get('motion')
                        if ps:keys.append((motion.line(view.pos[u],ps[0].point(0.)),*ps,motion.line(ps[-1].point(ps[-1].end),view.pos[v])))
                    paths=tuple(keys) or motion.edge_paths(view,u,v,kind)
                else:paths=motion.edge_paths(view,u,v,kind)
                guard_box(u,v,kind,paths)
        for u,v,*_ in lifts:
            if region[u] and region[v]:guard_box(u,v,'lift',motion.edge_paths(view,u,v,'lift') or ((motion.line(view.pos[u],view.pos[v]),),))
    sg=coarse.StateGraph(rel,view,region,tk,old.tcost,lifts,config,operate_cost,links=links,guards=guards,
                         skip=skip,launch_eligible=eligible)
    sg.loose_boxes=boxes
    sg.box_edge_guards={k:v for k,v in guards.items() if (v[0]|v[1])>>len(old.mechs)}
    sg.box_edge_blocked=sorted(skip)

    sg.bridges=old.bridges;sg.floats=old.floats;sg.timed=old.timed
    sg.pendulums=[dict(rod,blocked=[(mapping[u],mapping[v]) for u,v in rod['blocked'] if u in mapping and v in mapping])
                  for rod in old.pendulums]
    sg.float_row_names=old.float_row_names
    seen=set(); sg.mechanism_route_anchors={(int(sg.loc[a]),int(sg.loc[b]),m,t,s):v for (a,b,m,t,s),v in box_anchors.items()}
    for (u,v,c,m,typ,side),kind in zip(old._cond,old._cond_kind):
        a,b=loc(u),loc(v)
        if a is None or b is None:
            continue
        a,b=int(sg.loc[a]),int(sg.loc[b]); sig=(a,b,m,typ,side,kind)
        if min(a,b)<0 or sig in seen:
            continue
        seen.add(sig)
        sg.mechanism_route_anchors[a,b,m,typ,side]=(host.pos[int(old.glob[u])],host.pos[int(old.glob[v])])
        sg.cfwd[a].append((b,c,kind,m,typ,side));sg._cond.append((a,b,c,m,typ,side));sg._cond_kind.append(kind)
    sg._arrays()
    kinds=defaultdict(set)
    for u,v in mapping.items():
        if hasattr(old,'tk'):
            kinds[v].add(int(old.tk[old.loc[u]]))
    sg.mechanism_bridge=dict(mapped_host_anchors=len(mapping),host_anchors=len(old.glob),
                             mixed_transformer_nodes=sorted(u for u, values in kinds.items() if len(values)>1),
                             translated_conditional_edges=len(seen),translated_fan_edges=len(lifts))
    return sg
