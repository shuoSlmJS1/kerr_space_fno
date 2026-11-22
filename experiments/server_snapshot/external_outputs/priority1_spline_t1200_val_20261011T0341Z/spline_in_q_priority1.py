"""Frozen natural cubic interpolation in Q with two-knot linear tails."""
import numpy as np
from scipy.interpolate import CubicSpline


class SplineInQ:
    def __init__(self,q,xyz):
        q=np.asarray(q,dtype=np.float64)
        xyz=np.asarray(xyz,dtype=np.float64)
        if q.ndim!=1 or xyz.ndim!=3 or xyz.shape[0]!=len(q) or xyz.shape[-1]!=3 or len(q)<2:
            raise ValueError('Expected Q[N] and raw xyz[N,T,3], with at least two knots')
        if not np.isfinite(q).all() or not np.isfinite(xyz).all():
            raise ValueError('Nonfinite training data')
        # 保留原始行号，稳定排序；重复 Q 不允许静默合并。
        self.order=np.argsort(q,kind='stable')
        self.q=q[self.order]
        self.xyz=xyz[self.order]
        if not np.all(np.diff(self.q)>0):raise ValueError('Training Q must be unique')
        self.spline=CubicSpline(self.q,self.xyz,axis=0,bc_type='natural',extrapolate=False)
        self.left_slope=(self.xyz[1]-self.xyz[0])/(self.q[1]-self.q[0])
        self.right_slope=(self.xyz[-1]-self.xyz[-2])/(self.q[-1]-self.q[-2])

    def predict(self,q):
        q=np.asarray(q,dtype=np.float64)
        if q.ndim!=1 or not np.isfinite(q).all():raise ValueError('Invalid query Q')
        left=q<self.q[0];right=q>self.q[-1];inside=~(left|right)
        result=np.empty((len(q),*self.xyz.shape[1:]),dtype=np.float64)
        result[inside]=self.spline(q[inside])
        # 凸包外严格使用最近两个训练节点的割线，不使用样条自身的多项式外推。
        result[left]=self.xyz[0]+(q[left]-self.q[0])[:,None,None]*self.left_slope
        result[right]=self.xyz[-1]+(q[right]-self.q[-1])[:,None,None]*self.right_slope
        return result,inside,left,right


def unit_tests():
    q=np.array([2.,0.,3.,1.]);grid=np.arange(4.)
    truth=2*q[:,None,None]+grid[None,:,None]+np.arange(3.)[None,None,:]
    model=SplineInQ(q,truth)
    query=np.array([-1.,0.,.5,2.5,3.,4.])
    result,inside,left,right=model.predict(query)
    expected=2*query[:,None,None]+grid[None,:,None]+np.arange(3.)[None,None,:]
    assert np.allclose(result,expected,atol=1e-12,rtol=0)
    assert model.order.tolist()==[1,3,0,2]
    assert inside.tolist()==[False,True,True,True,True,False]
    assert left.sum()==right.sum()==1
    for bad in [np.array([1.,1.,2.,3.]),np.array([0.,1.,2.,np.nan])]:
        try:SplineInQ(bad,truth)
        except ValueError:pass
        else:raise AssertionError('Invalid training Q accepted')
    try:model.predict(np.array([np.nan]))
    except ValueError:pass
    else:raise AssertionError('Invalid query accepted')
    assert np.allclose(model.predict(q)[0],truth,atol=1e-12,rtol=0)
    return {'status':'PASS','cases':['stable_sort','inside_interpolation','left_secant_tail','right_secant_tail',
                                  'endpoint_membership','duplicate_rejection','nonfinite_training_rejection',
                                  'nonfinite_query_rejection','training_node_reconstruction']}
