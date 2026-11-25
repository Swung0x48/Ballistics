"""Joint runway families for given seam times (no state stepping).

DERIVED: every held world key contributes a closed-form position and speed
weight. At a concave seam affine transport and all prior finite-edge
constraints act on the same key variables. Free seam times, convex rim
transport and intermediate obstacle clearance are not certified by this LP.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from . import closedform as cf, physics as ph, supportcarry as sc


@dataclass(frozen=True)
class Family:
    center: np.ndarray  # p, v
    image: np.ndarray  # shared world-key variables, two per integer tick
    equal: np.ndarray
    equal_rhs: np.ndarray
    upper: np.ndarray
    upper_rhs: np.ndarray
    tick: float = 0.

    @classmethod
    def start(cls, reach, horizon):
        n = 2*math.ceil(horizon)
        return cls(np.r_[reach.origin,reach.velocity],np.zeros((6,n)),
                   np.zeros((0,n)),np.zeros(0),np.zeros((0,n)),np.zeros(0))

    def advance(self, reach, end):
        """Analytic weights for intersections of held inputs with a plane leg."""
        n = self.image.shape[1]//2
        if not self.tick<=end<=n:
            raise ValueError('plane leg outside scheduled horizon')
        d,S,T = cf._coeff(end-self.tick,reach.loss)
        p,v = self.center.reshape(2,3)
        M,N = self.image.reshape(2,3,-1)
        first = np.arange(n)
        left = np.maximum(self.tick,first)
        right = np.minimum(end,first+1)
        valid = left<right
        # Geometric prefix differences after factoring the decay at the end
        # of the overlap. Avoid subtracting nearly identical speed weights.
        D,A,B = cf._coeff(np.where(valid,right-left,0.),reach.loss)
        Q,U,_ = cf._coeff(np.where(valid,end-right,0.),reach.loss)
        C = B+U*A
        dp = (reach.axes[:,None,:]*(ph.DT*C)[None,:,None]).reshape(3,-1)
        dv = (reach.axes[:,None,:]*(Q*A)[None,:,None]).reshape(3,-1)
        return replace(self,center=np.r_[p+ph.DT*(S*v+T*reach.gravity),d*v+S*reach.gravity],
                       image=np.vstack((M+ph.DT*S*N+dp,d*N+dv)),tick=end)

    def on_edge(self, a, b):
        """Condition on one finite edge, in the XZ chart of the plane."""
        a,b = np.asarray(a)[[0,2]],np.asarray(b)[[0,2]]
        edge = b-a; length = np.linalg.norm(edge)
        p,M = self.center[[0,2]],self.image[[0,2]]
        if not length:
            return replace(self,equal=np.vstack((self.equal,M)),equal_rhs=np.r_[self.equal_rhs,a-p])
        unit = edge/length; side = np.array([-unit[1],unit[0]])
        along = unit@M; base = (p-a)@unit
        return replace(self,equal=np.vstack((self.equal,side@M)),equal_rhs=np.r_[self.equal_rhs,side@(a-p)],
                       upper=np.vstack((self.upper,along,-along)),upper_rhs=np.r_[self.upper_rhs,length-base,base])

    def transfer(self, old_normal, new_patch, radius):
        """DERIVED: subtract old skin offset, project foot and velocity."""
        n = new_patch.normal; P = np.eye(3)-np.outer(n,n)
        p,v = self.center.reshape(2,3); M,N = self.image.reshape(2,3,-1)
        return replace(self,center=np.r_[P@(p-radius*old_normal)+(new_patch.offset+radius)*n,P@v],
                       image=np.vstack((P@M,P@N)))

    def solve(self, direction=None, *, velocity=None, integer=False):
        """LP support or optional discrete-key candidate. Not a clearance proof."""
        obj = np.zeros(6) if direction is None else np.asarray(direction)
        E,Y = self.equal,self.equal_rhs
        if velocity is not None:
            E = np.vstack((E,self.image[[3,5]])); Y = np.r_[Y,np.asarray(velocity)[[0,2]]-self.center[[3,5]]]
        c = -(obj@self.image)
        if integer:
            answer = milp(c,integrality=np.ones(len(c)),bounds=Bounds(-np.ones(len(c)),np.ones(len(c))),
                          constraints=[LinearConstraint(E,Y,Y),LinearConstraint(self.upper,-np.inf,self.upper_rhs)])
        else:
            answer = linprog(c,A_ub=self.upper,b_ub=self.upper_rhs,A_eq=E,b_eq=Y,bounds=(-1,1),method='highs')
        if not answer.success:
            return None
        x = np.rint(answer.x) if integer else answer.x
        return self.center+self.image@x, x.reshape(-1,2)


def literal_schedule(schedule):
    """DERIVED candidate via rounded cumulative key sums at integer PSI.

    The integer prefix error is at most 1/2 per axis, so successive
    differences are in {-1,0,1}; this is not a pass/fail trajectory test.
    Integer PSI variables are used only to construct the actual key table.
    An incomplete final tick stays under its last key to the next PSI.
    """
    if not schedule:
        return ()
    ends = np.asarray([t for t,_ in schedule]); key = np.asarray([k for _,k in schedule])
    if np.any(np.diff(np.r_[0,ends])<=0) or np.any(np.abs(key)>1+1.e-8):
        raise ValueError('invalid fractional key table')
    count = math.ceil(ends[-1]); ends[-1] = count
    starts = np.r_[0.,ends[:-1]]
    time = np.arange(count+1)[:,None]
    area = np.clip(time-starts,0.,ends-starts)@key
    # Round-half-up rather than bankers' rounding, which can jump by two.
    keys = np.diff(np.floor(area+.5).astype(int),axis=0)
    stops = np.r_[1+np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1)),count]
    return tuple((int(t), keys[t-1]) for t in stops)
