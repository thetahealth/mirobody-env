"""Perfect-split audit bounds must work for pools larger than a floating-point integer."""
from fractions import Fraction
from math import comb
from haenv.world.validation.gates_shortcut import _split_p

def test_large_pool_perfect_split_is_a_finite_probability():
    labels=[0]*750+[1]*750
    got=_split_p(list(range(1500)),labels,1.0)
    assert 0.0<=got<1e-100
    assert got==float(Fraction(2,comb(1500,750)))

def test_small_pool_exact_bound_is_unchanged():
    labels=[0]*20+[1]*20
    assert _split_p(list(range(40)),labels,1.0)==2.0/comb(40,20)

def test_degenerate_labels_still_have_probability_one():
    assert _split_p([1]*1500,[1]*1500,1.0)==1.0
