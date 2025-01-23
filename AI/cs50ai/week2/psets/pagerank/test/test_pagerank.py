from pagerank import *
#import pytest_lazyfixture
#import pytest 

def test_transition():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"}
    }
    page = "1.html"
    damping_factor = 0.85
    result = transition_model(corpus, page, damping_factor)
    assert result == {
        "1.html": 0.05,
        "2.html": 0.475,
        "3.html": 0.475
    }
def test_transition_nolink():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"},
        "4.html": {}
    }
    page = "4.html"
    damping_factor = 0.85
    result = transition_model(corpus, page, damping_factor)
    assert result == {
        "1.html": 0.25,
        "2.html": 0.25,
        "3.html": 0.25,
        "4.html": 0.25
    }

def test_sample_pagerank():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"},
        "4.html": {}
    }
    damping_factor = 0.85
    samples = 100
    pagecount = sample_pagerank(corpus, damping_factor, samples)
    assert round(sum(pagecount.values())) == 1

def test_iterate():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"},
        "4.html": {}
    }
    damping_factor = 0.85
    pagecount = iterate_pagerank(corpus, damping_factor)
    assert sum(pagecount.values()) == 1
