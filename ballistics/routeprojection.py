"""Project shared evidence onto the terrain routing graph, never its outer candidates."""
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from . import motion, physics as ph, supportcarry
from .conformalsearch import Rail

XZ = [0, 2]
ANCHOR_REACH = 3.0  # existing landing-place mapping reach, not contact tolerance


class Anchors:
    def __init__(self, g):
        self.g = g
        self.polygons = {}
        roads, rails = defaultdict(list), defaultdict(list)
        for i, patches in enumerate(g.info['terrain_support']):
            for k in patches:
                roads[k].append(i)
        for i in np.flatnonzero(g.lane >= 0):
            rails[int(g.lane[i])].append(int(i))
        self.roads = {k: (ids, cKDTree(g.pos[ids])) for k, ids in roads.items()}
        self.rails = {k: (ids, cKDTree(g.pos[ids])) for k, ids in rails.items()}

    def road(self, patch, pos, radius, contact=None):
        if patch not in self.roads:
            return -1, 'patch_has_no_coarse_place'
        ids, tree = self.roads[patch]
        local = tree.query_ball_point(pos, ANCHOR_REACH)
        if not local:
            return -1, 'no_local_place_on_patch'
        p = self.g.info['conformal'].patches[patch]
        # Compare feet on the plane, not sphere centres in XZ on a slope.
        foot = np.asarray(contact)[XZ] if contact is not None else (pos - radius * p.normal)[XZ]
        if patch not in self.polygons:
            self.polygons[patch] = p.polygon.buffer(1.e-7)
        polygon = self.polygons[patch]
        for i in sorted((ids[j] for j in local), key=lambda i: (np.linalg.norm(self.g.pos[i] - pos), i)):
            a = self.g.pos[i]
            q = (a - (a @ p.normal - p.offset) * p.normal)[XZ]
            if polygon.covers(Point(foot) if np.linalg.norm(q-foot) < 1.e-7 else LineString((foot, q))):
                return i, 'same_finite_patch'
        return -1, 'patch_anchor_chord_leaves_polygon'

    def rail(self, node, pos):
        if not isinstance(node, Rail) or node.chain not in self.rails:
            return -1, 'rail_chain_has_no_coarse_place'
        if node.line != 'groove' and self.g.lanes[node.chain].get('double'):
            return -1, 'no_coarse_place_on_crown_line'
        ids, tree = self.rails[node.chain]
        d, j = tree.query(pos)
        if d > ANCHOR_REACH:
            return -1, 'no_local_place_on_rail_chain'
        return ids[j], 'same_rail_chain_and_line'


def attach(tf, g, *, flights=True, rolling=True, retained=True):
    """Publish before mechanism screening; leave original terrain rolls intact.

    `launch` is the existing trajectory-backed transport slot. Its display
    type is separately recorded and it does not enlarge rolling SCCs.
    """
    shared = g.info['shared_motion_graph']
    lookup = Anchors(g)
    launches = g.info.setdefault('launches', {})
    groups = defaultdict(list)
    audit = []
    counts = Counter()
    balls = {b.name: b for b in ph.BALLS.values()}
    seen = set()

    def publish(src, dst, row, status, kind):
        if id(row) in seen:
            return
        seen.add(id(row))
        a, b = row['source_patch'], row['target_patch']
        origin, end = (np.asarray(row[k],float) for k in ('origin','position'))
        radius = balls[src.ball].radius + supportcarry.SKIN
        def endpoint(node, patch, pos):
            return lookup.road(patch, pos, radius) if patch >= 0 else lookup.rail(node, pos)
        u, why = endpoint(src, a, origin)
        contact = next((c['point'] for c in reversed(row.get('contacts',()))
                        if c.get('kind') == 'land' and 'point' in c and
                        c.get('triangle') == row.get('face') and np.linalg.norm(np.asarray(c['position'])-end) < 1.e-7),None)
        v, why2 = lookup.road(b,end,radius,contact) if b>=0 else endpoint(dst,b,end)
        outcome = 'mapped' if min(u, v) >= 0 and u != v else 'same_place' if u == v >= 0 else 'unmapped'
        counts[kind, outcome] += 1
        audit.append(dict(ball=src.ball, source=src, target=dst, patch_pair=(a,b), anchors=(u,v),
                          source_reason=why, target_reason=why2, status=outcome, kind=kind,
                          direction=src.direction if isinstance(src,Rail) else None,
                          evidence=status, origin=row['origin'], position=row['position']))
        if outcome != 'mapped' or not row.get('motion'):
            return
        ticks = row.get('ticks', np.linalg.norm(end-origin) / (tf.FOLLOW_SPEED * ph.DT))
        velocity = row.get('velocity', row.get('origin_velocity', np.zeros(3)))
        start = np.asarray(row.get('launch', origin),float)
        keys = row.get('key_table', ())
        groups[u,v].append(dict(site=u, ball=src.ball, ticks=ticks, time=ticks*ph.DT,
            flight_ticks=row.get('flight_ticks',0.), verified=not status.startswith('retained'),
            witness_status=status, route_type=kind, representative=row.get('mode',kind),
            retained_reference=row.get('retained_reference',row.get('witness_reference')),
            support_patches=(a,b), shared_supported_transfer=True,
            launch_position=start, launch_velocity=velocity, launch_speed=float(np.linalg.norm(velocity)),
            landing_position=end, landing_face=row.get('face',-1), first_landing_node=v,
            motion=row['motion'], contacts=row.get('contacts',()),
            projection_chords=(motion.line(g.pos[u],origin),motion.line(end,g.pos[v])),
            way=np.vstack((g.pos[u],row['origin'],*(p.point(p.end) for p in row['motion']),g.pos[v])),
            across=float(np.linalg.norm((end-start)[XZ])), down=float(start[1]-end[1]),
            rail_parts=row.get('rail_parts',()), key_table=keys, keys_in_air=keys[-1][1] if keys else np.zeros(2),
            air_inputs=tuple(x[1] for x in keys), switch_tick=keys[0][0] if len(keys)>1 else 0,
            runup=row.get('runup',0.), rim_ticks=row.get('rim_ticks',0.), bounce_count=0,rail_support_prefix_ticks=0.))

    if flights:
        for (a,b), rows in shared['witnesses'].items():
            for row in rows:
                kind = ('shared_rail' if row.get('rail_parts') else 'convex_crest_flight'
                        if row.get('mode','').startswith('convex') else 'shared_flight')
                publish(a,b,row,'literal_confirmed',kind)
    for rec in (g.info.get('portal_verdicts', ()) if rolling else ()):
        row = rec.get('witness')
        if rec['status'] == 'verified' and row is not None and row.get('kind') == 'roll':
            publish(rec['source'],rec['target'],row,
                    'quasistatic_certificate' if row.get('quasistatic') else 'literal_confirmed',
                    'quasistatic_roll' if row.get('quasistatic') else 'shared_supported_roll')
    for (a,b),row in (shared.get('retained_rolling',{}).items() if retained else ()):
        publish(a,b,row,'retained_verified_roll','shared_supported_roll')
    for (u,v), rows in groups.items():
        before = tf.variants(g,u,v)
        es = tuple(sorted((*before,*rows), key=lambda e: (e.get('witness_status','').startswith('retained'), e['ticks'])))
        launches[u,v] = dict(es[0],variants=es)
        g.adj[u] = [x for x in g.adj[u] if not (x[0] == v and x[2] == 'launch')]
        g.adj[u].append((v,tf.cost(es[0]),'launch'))
    old = g.info.setdefault('route_projection', [])
    old.extend(audit)
    return {f'{kind}:{state}': n for (kind,state),n in counts.items()}
