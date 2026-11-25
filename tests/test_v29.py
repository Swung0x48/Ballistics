from types import SimpleNamespace
import numpy as np

from ballistics import closedform as cf, physics as ph, railoutlet, railride, sectionroutes, sharedrail


def test_midrail_speed_band_selects_absolute_integer_key_switch():
    first=cf.RollingAxis(.08,.01); second=cf.RollingAxis(.13,.01)
    v,d,start=5.,18.,7.4
    a,b=sorted((first.over(v,d)[0],second.over(v,d)[0])); wanted=(a+(b-a)/4,b-(b-a)/4)
    tick,arc,_,out=railoutlet.midrail_switch(first,second,v,d,wanted,start)
    x,w=first.at(v,tick-start)
    assert isinstance(tick,int) and 0.<arc[0]<x<arc[1]<d
    assert wanted[0]<out<wanted[1]
    assert np.isclose(out,second.over(w,d-x)[0])


def test_checkpoint_cannot_replace_the_arriving_ball():
    ok=dict(nodes=[1,2],edges=[{'status':'verified'}]); a={('wood','stone'):ok}; b={('paper','paper'):ok}
    result=sectionroutes.link_kinds((a,b),'wood')
    assert result['first_missing_sector']==2 and result['boundary_states'][-1]['reachable_entry_kinds']==['stone']
    result=sectionroutes.link_kinds((a,{('stone','wood'):ok}),'wood')
    assert [(r['entry_ball'],r['exit_ball']) for r in result['checked_prefix']]==[('wood','stone'),('stone','wood')]


def test_rail_continuity_requires_distinct_coincident_forward_support():
    p=np.array([0.,2.,0.]); v=np.array([4.,0.,0.]); body=ph.State(p,v)
    a=SimpleNamespace(double=True,axis=None,tangent=np.array([[1.,0.,0.]])); paths=SimpleNamespace(lanes={0:a,1:a}, geometry_support=lambda *_:5)
    old=railride.Catch(0,0,1,body,0.,3); nxt=railride.Catch(1,0,1,body,0.,5,rebound=body)
    dep=SimpleNamespace(kind='chain_end',body=body,source=old)
    result=sharedrail._rail_join(paths,ph.WOOD,dep,dict(catch=nxt))
    assert result['ticks']==0. and result['catch'].rebound is None
    assert np.array_equal(result['catch'].body.vel,v)
    assert sharedrail._rail_join(paths,ph.WOOD,dep,dict(catch=old)) is None
    far=railride.Catch(1,0,1,ph.State(p+[0.,0.,.5],v),0.,5)
    assert sharedrail._rail_join(paths,ph.WOOD,dep,dict(catch=far)) is None
