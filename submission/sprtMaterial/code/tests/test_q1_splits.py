import numpy as np
import pytest
from src.q1.data import load_q1
from src.q1.splits import inner_splits, outer_splits, paper_split


@pytest.fixture(scope="module")
def setup():
    d = load_q1()
    groups = d.targets.date_group
    return groups, d.style, outer_splits(groups, d.style)


def test_outer_each_beer_tested_once_and_groups_isolated(setup):
    groups, style, splits = setup
    n = len(style)
    assert len(splits) == 5 and all(len(rep) == 5 for rep in splits)
    for rep in splits:
        counts = np.zeros(n, int)
        for tr, te in rep:
            counts[te] += 1
            assert not set(groups[tr]) & set(groups[te])
            assert len(np.intersect1d(tr, te)) == 0
            assert len(tr) + len(te) == n
        assert (counts == 1).all()


def test_outer_deterministic_and_repeats_differ(setup):
    groups, style, splits = setup
    again = outer_splits(groups, style)
    for a, b in zip(splits, again):
        for (tra, tea), (trb, teb) in zip(a, b):
            assert np.array_equal(tra, trb) and np.array_equal(tea, teb)
    tests0 = [set(te) for _, te in splits[0]]
    tests1 = [set(te) for _, te in splits[1]]
    assert sorted(map(sorted, tests0)) != sorted(map(sorted, tests1))


def test_inner_splits(setup):
    groups, style, splits = setup
    train_idx = splits[0][0][0]
    inner = inner_splits(train_idx, groups, style, seed=100)
    assert len(inner) == 4
    vals = []
    for a, b in inner:
        assert set(a) <= set(train_idx) and set(b) <= set(train_idx)
        assert not set(groups[a]) & set(groups[b])
        vals.append(b)
    allv = np.concatenate(vals)
    assert len(allv) == len(set(allv))
    assert set(allv) == set(train_idx)


def test_paper_split(setup):
    _, style, _ = setup
    tr, te = paper_split(style)
    assert len(te) == 75
    assert len(np.intersect1d(tr, te)) == 0
    assert len(tr) + len(te) == 250
