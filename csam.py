import numpy as np
from scipy.sparse import csr_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSAM:
    def __init__(self, feature_dimension, layer_importance = [1, 1.2, 1.4, 1.6], keyword_importance = 0.2, edge_importance = 0.1, min_layer_edge_context = 2):
        """
        Initializes the Contextualized Sparse Attention Mechanism.
        Args:
            feature_dimension (int): The number of features in the sparse representations
            layer_importance (list): A list of floats that will be used to determine how important the specific layer is.
                                     The index in the list represents the layer.
            keyword_importance (float): A float value of how important a keyword is in determining an attention score.
            edge_importance (float): A float value on how important an edge is in determining the attention score
            min_layer_edge_context (int): The minimum layer at which to incorporate edge context
        """
        self.feature_dimension = feature_dimension
        self.layer_importance = layer_importance
        self.keyword_importance = keyword_importance
        self.edge_importance = edge_importance
        self.min_layer_edge_context = min_layer_edge_context
        logging.info("CSAM initialized")

    def attend(self, query_chunk, hkg_nodes, hkg_graph):
        """
        Calculates attention scores between a query and the nodes in the HKG.

        Args:
            query_chunk (dict): A dictionary representing the sparse
                                       representation of the information.
            hkg_nodes (list):  A list of tuples, where each tuple contains
                                   (node_id, node_data) from HKG-AG.
            hkg_graph (networkx.MultiDiGraph): A graph object that contains the edges of the HKG_AG
        Returns:
            dict: A dictionary mapping node_id to the attention score.
        """
        logging.info(f"Querying with: {query_chunk}")
        try:
            sparse_query = self._dict_to_sparse(query_chunk)
        except ValueError as e:
            logging.error(f"Error in attend method: {e}")
            return {}
        attention_scores = {}

        for node_id, node_data in hkg_nodes:
             if node_data.get('sanm_references'):
                total_similarity = 0
                for sanm_ref in node_data['sanm_references']:
                    sparse_mem = self._array_to_sparse(sparse_query.toarray().flatten() * sanm_ref)
                    similarity = self._calculate_similarity(sparse_query, sparse_mem)
                    total_similarity += similarity

                if len(node_data['sanm_references']) > 0:
                     average_similarity = total_similarity/len(node_data['sanm_references'])
                else:
                     average_similarity = 0

                layer_multiplier = self.layer_importance[node_data['layer']] if node_data['layer'] < len(self.layer_importance) else 1.0

                keyword_score = 0
                if node_data['layer'] >= 2 and 'data' in node_data and 'description' in node_data['data']:
                     for keyword, _ in query_chunk.items():
                         if str(keyword) in node_data['data']['description']:
                             keyword_score += self.keyword_importance

                edge_score = 0
                if node_data['layer'] >= self.min_layer_edge_context:
                     for source_id, target_id, key, data in hkg_graph.edges(keys = True, data=True):
                         if source_id == node_id and data['relation'] == "is_a":
                             target_node_data = hkg_graph.nodes.get(target_id, {})
                             if 'data' in target_node_data and 'description' in target_node_data['data']:
                                 for keyword, _ in query_chunk.items():
                                     if str(keyword) in target_node_data['data']['description']:
                                         edge_score += self.edge_importance

                attention_score = average_similarity * layer_multiplier + keyword_score + edge_score
                attention_scores[node_id] = attention_score
                logging.debug(f"  Node: {node_id}, Similarity: {average_similarity:.2f}, Layer: {node_data['layer']}, Layer Multiplier: {layer_multiplier:.2f}, Keyword Score: {keyword_score:.2f}, Edge Score: {edge_score:.2f}, Attention Score: {attention_score:.2f}")
             else:
                attention_scores[node_id] = 0
                logging.debug(f"  Node: {node_id}, No sanm_reference, Attention Score: 0.00")

        return attention_scores

    def _dict_to_sparse(self, information_chunk):
        """Converts a dictionary to a sparse matrix"""
        indices = []
        data = []
        for feature, value in information_chunk.items():
            try:
                index = int(feature)
                indices.append(index)
                data.append(value)
            except ValueError:
                logging.warning(f"Skipping non-integer feature key: '{feature}'")
        return csr_matrix((data, ([0] * len(data), indices)), shape=(1, self.feature_dimension))

    def _array_to_sparse(self, array):
        """Converts an array to a sparse matrix"""
        data = []
        indices = []
        for index, value in enumerate(array):
            if value != 0:
                data.append(value)
                indices.append(index)

        return csr_matrix((data, ([0] * len(data), indices)), shape=(1, len(array)))

    def _calculate_similarity(self, chunk1, chunk2):
        """Calculates the cosine similarity between two sparse matrices."""
        dot_product = chunk1.dot(chunk2.T).toarray()[0][0]
        magnitude1 = np.sqrt(chunk1.dot(chunk1.T).toarray()[0][0])
        magnitude2 = np.sqrt(chunk2.dot(chunk2.T).toarray()[0][0])

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity

if __name__ == "__main__":
     # Example Usage
    from hkg_ag import HKG_AG

    feature_dimension = 100
    csam = CSAM(feature_dimension=feature_dimension)
    hkg = HKG_AG()

    # Mock HKG nodes for testing
    mock_hkg_nodes = [(node_id, data) for node_id, data in hkg.graph.nodes(data=True)]

    # Mock Query
    mock_query = {"1": 0.9, "3": 0.7, "90": 0.8, "91": 0.5}

    # Run the CSAM
    attention_scores = csam.attend(mock_query, mock_hkg_nodes, hkg.graph)
    print("\nAttention Scores:")
    for node_id, score in attention_scores.items():
         print(f"  Node: {node_id}, Score: {score:.2f}")