import pytest
from nsil import NSIL
from unittest.mock import MagicMock
import numpy as np
from scipy.sparse import csr_matrix

@pytest.fixture
def nsil_instance():
    return NSIL(feature_dimension=100)

def test_dict_to_sparse_valid_keys_nsil(nsil_instance):
    chunk = {"1": 0.5, "2": 0.8, "99": 0.2}
    sparse_matrix = nsil_instance._dict_to_sparse(chunk)
    assert sparse_matrix.shape == (1, 100)
    assert sparse_matrix[0, 1] == 0.5
    assert sparse_matrix[0, 2] == 0.8
    assert sparse_matrix[0, 99] == 0.2

def test_dict_to_sparse_invalid_keys_nsil(nsil_instance, caplog):
    chunk = {"1": 0.5, "invalid": 0.8, "99": 0.2, "also_invalid": 0.1}
    sparse_matrix = nsil_instance._dict_to_sparse(chunk)
    assert sparse_matrix.shape == (1, 100)
    assert sparse_matrix[0, 1] == 0.5
    assert sparse_matrix[0, 99] == 0.2
    assert "Skipping non-integer feature key: 'invalid'" in caplog.text
    assert "Skipping non-integer feature key: 'also_invalid'" in caplog.text

def test_array_to_sparse_nsil(nsil_instance):
    array = [0, 1, 0, 2, 0]
    sparse_matrix = nsil_instance._array_to_sparse(array)
    assert sparse_matrix.shape == (1, len(array))
    assert sparse_matrix[0, 1] == 1
    assert sparse_matrix[0, 3] == 2

def test_integrate_basic(nsil_instance):
    query_chunk = {"1": 0.9, "3": 0.7}
    attention_scores = {"node1": 0.8, "node2": 0.6}
    mock_sanm = MagicMock()
    mock_hkg = MagicMock()
    mock_hkg.get_node.return_value = {'sanm_references': [np.array([0.1] * 100)]}
    mock_hkg.graph.edges.return_value = []  # No rules applied in this basic test

    combined_vector, updated_attention = nsil_instance.integrate(query_chunk, attention_scores, mock_sanm, mock_hkg)

    assert isinstance(combined_vector, csr_matrix)
    assert updated_attention == attention_scores  # Basic test, no rules to change scores

def test_integrate_rule_boost(nsil_instance):
    nsil_instance.rule_importance = 0.5
    query_chunk = {"1": 1.0}
    attention_scores = {"top_node": 0.5, "node2": 0.7}
    mock_sanm = MagicMock()
    mock_hkg = MagicMock()
    mock_hkg.get_node.side_effect = [
        {'sanm_references': [np.array([0.1] * 100)]},  # For top node
        {'data': {'description': 'This node contains 1'}}, # For target node of the rule
    ]
    mock_hkg.graph.edges.return_value = [("top_node", "node2", None, {'relation': 'is_a'})]

    combined_vector, updated_attention = nsil_instance.integrate(query_chunk, attention_scores, mock_sanm, mock_hkg)

    assert updated_attention["node2"] > 0.7 # Score should be boosted
    
def test_integrate_no_top_nodes(nsil_instance):
    query_chunk = {"1": 0.9, "3": 0.7}
    attention_scores = {"node1": 0.8, "node2": 0.6}
    mock_sanm = MagicMock()
    mock_hkg = MagicMock()
    mock_hkg.get_node.return_value = None # Simulate no SANM references

    combined_vector, updated_attention = nsil_instance.integrate(query_chunk, attention_scores, mock_sanm, mock_hkg)

    assert combined_vector.nnz == 0 # No SANM references to combine
    assert updated_attention == attention_scores