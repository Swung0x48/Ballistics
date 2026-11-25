"""Trusted-local JSON archive for retaining, not re-certifying, old witnesses."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import motion, physics as ph


def archive_for(info, explicit=None):
    if explicit is not None:return Path(explicit)
    name=Path(info.get('map','')).stem
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):return None
    return Path('out/regions-takeoff')/(name+'-retained-witnesses.json')


def priority_points(info):
    if 'target_priorities' in info:return info['target_priorities']
    p=archive_for(info)
    if p is None:return ()
    q=p.with_name(p.name.replace('-retained-witnesses','-target-priorities'))
    if not q.is_file():return ()
    data=json.loads(q.read_bytes())
    if data['map']!=info.get('map'):raise ValueError('target priorities map identity mismatch')
    return data['points']


def signature(pos):
    p=np.ascontiguousarray(pos,dtype='<f8')
    return dict(shape=list(p.shape),sha256=hashlib.sha256(p.tobytes()).hexdigest())


def version(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None


def encode_piece(p):
    return [p.kind,p.p,p.v,p.b,p.loss,p.end,p.arc]


def decode_piece(x):
    kind,p,v,b,loss,end,arc=x
    if kind not in ('line','rolling','air','arc'):raise ValueError('unknown archived geometry')
    if kind=='arc':arc=(*arc[:2],np.asarray(arc[2]),*arc[3:])
    return motion.Piece(kind,np.asarray(p),np.asarray(v),np.asarray(b),loss,end,tuple(arc))


def attach(tf,g,path,info=None):
    """Attach old graph alternatives before gate/transformer screening.

    No implication that a reference motion is a literal-key witness. Old
    cone-only proposals are archived as unresolved; they were not witnesses.
    A mismatched mapping rejects the ANCHOR, not the archived trajectory.
    """
    if path is None or not path.is_file():return dict(variants=0,playable=0,unresolved=0,pairs=0)
    data=json.loads(path.read_bytes())
    if data['map']!=(info or {}).get('map') or data['mapping']!=signature(g.pos):
        raise ValueError('retained witness archive does not match the coarse mapping')
    launches=g.info['launches']; groups=defaultdict(list); unresolved=[]; archived=[]
    balls={b.name for b in ph.BALLS.values()}
    for row in data['variants']:
        u,v=row['from'],row['to']; e=dict(row['edge'],witness_status='retained_legacy',literal_keys_rechecked=False,verified=False,
                                        retained_reference=row.get('version',data['version']))
        if not (0<=u<len(g.pos) and 0<=v<len(g.pos) and e['ball'] in balls):raise ValueError('invalid retained witness endpoint')
        archived.append((u,v,e))
        if not row.get('was_witness') or not e['motion']:
            e['witness_status']='unresolved_old_candidate';unresolved.append((u,v,e));continue
        e['motion']=tuple(decode_piece(x) for x in e['motion'])
        for k in ('way','launch_position','launch_velocity','landing_position','keys_in_air'):
            e[k]=np.asarray(e[k])
        groups[u,v].append(e)
    for (u,v),es in groups.items():
        before=tf.variants(g,u,v)
        rows=tuple(sorted((*before,*es),key=lambda e:(e.get('witness_status')=='retained_legacy',e['ticks'],e['ball'])))
        launches[u,v]=dict(rows[0],variants=rows)
        # Replace only the launch edge for this mapping pair.
        g.adj[u]=[x for x in g.adj[u] if not (x[0]==v and x[2]=='launch')]
        g.adj[u].append((v,tf.cost(rows[0]),'launch'))
    g.info['retained_archive']=archived
    g.info['retained_unresolved']=unresolved
    return dict(variants=len(archived),playable=sum(map(len,groups.values())),unresolved=len(unresolved),pairs=len(groups))

def rolling_archive_for(info):
    path=archive_for(info)
    return None if path is None else path.with_name(path.name.replace('-retained-witnesses','-retained-rolling'))


def attach_rolling(g,path,info):
    """Restore previously checked local rolling edges without re-certifying them."""
    if path is None or not path.is_file():return dict(retained=0,rechecked=0)
    data=json.loads(path.read_bytes()); surface=g.info['conformal']; layout=g.info['conformal_search']
    if data['map']!=info.get('map') or data['geometry']!=signature(surface.triangles):
        raise ValueError('retained rolling archive does not match conformal source geometry')
    shared=g.info['shared_motion_graph']; old={}; checked=0
    for row in data['rows']:
        index=layout.patch_nodes[row['ball']]; a,b=(index[int(row[k])] for k in ('source_patch','target_patch'))
        if a==b or (a,b) in shared['rolling'] or (a,b) in shared['witnesses']:
            checked+=1;continue
        witness=dict(row,witness_status='retained_verified_roll',retained_reference=data['version'],
                     motion=tuple(decode_piece(x) for x in row['motion']))
        old[a,b]=witness
    shared['retained_rolling']=old
    return dict(retained=len(old),rechecked=checked)


def outlet_priorities(info):
    if 'rail_outlet_targets' in info:return info['rail_outlet_targets']
    archive=archive_for(info)
    if archive is None:return ()
    path=archive.with_name(archive.name.replace('-retained-witnesses','-rail-outlet-targets'))
    if not path.is_file():return ()
    data=json.loads(path.read_bytes())
    if data['map']!=info.get('map'):raise ValueError('rail outlet priorities map identity mismatch')
    return data['points']
