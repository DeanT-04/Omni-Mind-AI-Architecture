import pytest
from csam import CSAM
from unittest.mock import MagicMock
import numpy as np
from scipy.sparse import csr_matrix

@pytest.fixture
def csam_instance():
    return CSAM(feature_dimension=100)

def test_dict_to_sparse_valid_keys(csam_instance):
    chunk = {"1": 0.5, "2": 0.8, "99": 0.2}
    sparse_matrix = csam_instance._dict_to_sparse(chunk)
    assert sparse_matrix.shape == (1, 100)
    assert sparse_matrix[0, 1] == 0.5
    assert sparse_matrix[0, 2] == 0.8
    assert sparse_matrix[0, 99] == 0.2

def test_dict_to_sparse_invalid_keys(csam_instance, caplog):
    chunk = {"1": 0.5, "invalid": 0.8, "99": 0.2, "also_invalid": 0.1}
    sparse_matrix = csam_instance._dict_to_sparse(chunk)
    assert sparse_matrix.shape == (1, 100)
    assert sparse_matrix[0, 1] == 0.5
    assert sparse_matrix[0, 99] == 0.2
    assert "Skipping non-integer feature key: 'invalid'" in caplog.text
    assert "Skipping non-integer feature key: 'also_invalid'" in caplog.text

def test_array_to_sparse(csam_instance):
    array = [0, 1, 0, 2, 0]
    sparse_matrix = csam_instance._array_to_sparse(array)
    assert sparse_matrix.shape == (1, len(array))
    assert sparse_matrix[0, 1] == 1
    assert sparse_matrix[0, 3] == 2

def test_calculate_similarity(csam_instance):
    chunk1 = csr_matrix([[1, 0, 2]])
    chunk2 = csr_matrix([[1, 0, 0]])
    similarity = csam_instance._calculate_similarity(chunk1, chunk2)
    assert similarity == pytest.approx(0.447, 0.001)

    chunk3 = csr_matrix([[0, 0, 0]])
    similarity_zero = csam_instance._calculate_similarity(chunk1, chunk3)
    assert similarity_zero == 0.0

def test_attend_basic(csam_instance, caplog):
    query_chunk = {"10": 0.7, "20": 0.9}
    hkg_nodes = [
        ("node1", {"sanm_references": [np.array([0.1] * 100)], 'layer': 0}),
        ("node2", {"sanm_references": [np.array([0.9] * 100)], 'layer': 1})
    ]
    mock_graph = MagicMock()
    attention_scores = csam_instance.attend(query_chunk, hkg_nodes, mock_graph)
    assert "node1" in attention_scores
    assert "node2" in attention_scores

def test_attend_with_keyword_boost(csam_instance):
    csam_instance.keyword_importance = 0.5
    query_chunk = {"1": 0.8}
    hkg_nodes = [
        ("node1", {"sanm_references": [np.array([0.5] * 100)], 'layer': 2, 'data': {'description': 'This node contains 1'}}),
    ]
    mock_graph = MagicMock()
    attention_scores = csam_instance.attend(query_chunk, hkg_nodes, mock_graph)
    assert attention_scores["node1"] > 0  # Ensure keyword boost increases score

def test_attend_with_edge_boost(csam_instance):
    csam_instance.edge_importance = 0.3
    csam_instance.min_layer_edge_context = 0
    query_chunk = {"1": 0.9}
    hkg_nodes = [
        ("node1", {"sanm_references": [np.array([0.5] * 100)], 'layer': 0, 'data': {'description': 'parent description'}}),
        ("node2", {"sanm_references": [np.array([0.6] * 100)], 'layer': 1, 'data': {'description': 'This is 1'}}),
    ]
    mock_graph = MagicMock()
    mock_graph.edges.return_value = [("node2", "node1", None, {'relation': 'is_a'})]
    mock_graph.nodes = {"node1": {'data': {'description': 'parent description'}}} # Mock node data access
    attention_scores = csam_instance.attend(query_chunk, hkg_nodes, mock_graph)
    assert attention_scores.get("node2", 0) > 0 # Ensure edge boost increases score

def test_attend_no_sanm_references(csam_instance):
    query_chunk = {"1": 0.5}
    hkg_nodes = [("node1", {'sanm_references': [], 'layer': 0})]
    mock_graph = MagicMock()
    attention_scores = csam_instance.attend(query_chunk, hkg_nodes, mock_graph)
    assert attention_scores["node1"] == 0