from heredity import *
import pytest

def test_joint_probability():
    people = {
        'Harry': {'name': 'Harry', 'mother': 'Lily', 'father': 'James', 'trait': None},
        'James': {'name': 'James', 'mother': None, 'father': None, 'trait': True},
        'Lily': {'name': 'Lily', 'mother': None, 'father': None, 'trait': False},
    }
    one_gene = {'Harry'}
    two_genes = {'James'}
    have_trait = {'James'}
    assert joint_probability(people, one_gene, two_genes, have_trait) == 0.0026643247488