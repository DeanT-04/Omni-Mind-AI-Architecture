import numpy as np
from scipy.sparse import csr_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NSIL:
    def __init__(self, feature_dimension, rule_importance = 0.1):
        """
        Initializes the Neuro-Symbolic Integration Layer.
        Args:
            feature_dimension (int): The number of features in the sparse representations
             rule_importance (float): A float value on how much importance the rules have when calculating the attention.
        """
        self.feature_dimension = feature_dimension
        self.rule_importance = rule_importance
        self.rules = [
             {
                 "if_relation": "is_a",
                 "then_boost": True,
                  "target_attributes": ["description"]
             },
            {
                 "if_relation": "has_attribute",
                 "then_boost": True,
                 "target_attributes": ["location"]
            }
        ]
        logging.info("NSIL initialized")

    def integrate(self, query_chunk, attention_scores, sanm, hkg):
        """
        Integrates information from SANM, HKG-AG, and CSAM.

        Args:
            query_chunk (dict): A dictionary representing the sparse
                                       representation of the query.
            attention_scores (dict): A dictionary mapping node_id to the attention score.
            sanm (SANM): The SANM object.
            hkg (HKG_AG): The HKG_AG object.
        Returns:
            tuple: A tuple containing:
                csr_matrix: A combined sparse vector of the most relevant SANM references.
                dict: A dictionary mapping node_id to the updated attention scores.
        """
        logging.info("\n--- Integrating ---")
        # Sort nodes by attention score
        sorted_nodes = sorted(attention_scores.items(), key = lambda item: item[1], reverse = True)

        combined_vector = csr_matrix((1, self.feature_dimension))

        #Get top nodes with SANM references
        top_nodes = []
        for node_id, score in sorted_nodes:
            node_data = hkg.get_node(node_id)
            if node_data and node_data.get('sanm_references'):
                top_nodes.append(node_id)
                logging.debug(f"  Top Node: {node_id}, Score: {score:.2f}")
            if len(top_nodes) >= 5:
                break

        #Combine SANM References
        for node_id in top_nodes:
            node_data = hkg.get_node(node_id)
            if node_data and node_data.get('sanm_references'):
                for sanm_ref in node_data['sanm_references']:
                    sparse_vector = self._array_to_sparse(self._dict_to_sparse(query_chunk).toarray().flatten() * sanm_ref)
                    combined_vector += sparse_vector

        #Apply rules
        updated_attention_scores = attention_scores.copy()
        for node_id in top_nodes:
             node_data = hkg.get_node(node_id)
             if node_data:
                 for source_id, target_id, key, data in hkg.graph.edges(keys = True, data = True):
                      if source_id == node_id:
                           for rule in self.rules:
                                if data['relation'] == rule['if_relation']:
                                    if target_id in updated_attention_scores:

                                        keyword_score = 0
                                        target_node_data = hkg.get_node(target_id)
                                        if target_node_data and 'data' in target_node_data:
                                            for target_attribute in rule["target_attributes"]:
                                                if target_attribute in target_node_data['data']:
                                                  for keyword, _ in query_chunk.items():
                                                       if str(keyword) in target_node_data['data'][target_attribute]:
                                                          keyword_score += self.rule_importance
                                        if rule["then_boost"]:
                                          updated_attention_scores[target_id] += keyword_score
                                          logging.debug(f"  Rule applied, boosting node: '{target_id}' due to node: '{source_id}' having a '{rule['if_relation']}' relation, new score: {updated_attention_scores[target_id]:.2f}")
                                        else:
                                             updated_attention_scores[target_id] -= keyword_score
                                             logging.debug(f"  Rule applied, reducing node: '{target_id}' due to node: '{source_id}' having a '{rule['if_relation']}' relation, new score: {updated_attention_scores[target_id]:.2f}")
        logging.info("--- Integration Complete ---")
        return combined_vector, updated_attention_scores

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

if __name__ == "__main__":
    from sanm import SANM
    from hkg_ag import HKG_AG
    from csam import CSAM
    from data.mock_data import get_initial_data

    # Determine the feature dimension (max feature index + 1)
    initial_data = get_initial_data()
    feature_dimension = 0
    for data_chunk in initial_data:
      for feature in data_chunk:
          try:
              feature_dimension = max(feature_dimension, int(feature))
          except ValueError:
              logging.warning(f"Skipping non-integer feature key: '{feature}' during dimension calculation.")
    feature_dimension += 1

    # Initialize components
    sanm = SANM(feature_dimension=feature_dimension, similarity_threshold=0.6, num_trees=10)
    hkg = HKG_AG()
    csam = CSAM(feature_dimension=feature_dimension)
    nsil = NSIL(feature_dimension=feature_dimension)

    # Load initial SANM data
    print("\nLoading initial data for SANM:")
    for data_chunk in initial_data:
        sanm.add(data_chunk)
    sanm.save_index()

    # Mock Query
    mock_query = {"1": 0.9, "3": 0.7, "90": 0.8, "91": 0.8}

    # Mock Attention Scores
    mock_hkg_nodes = [(node_id, data) for node_id, data in hkg.graph.nodes(data=True)]
    attention_scores = csam.attend(mock_query, mock_hkg_nodes, hkg.graph)

    # Integrate
    integrated_vector, updated_attention_scores = nsil.integrate(mock_query, attention_scores, sanm, hkg)
    print(f"\nCombined Vector: {integrated_vector.toarray()}")
    print("\nFinal Attention Scores:")
    for node_id, score in updated_attention_scores.items():
         print(f"  Node: {node_id}, Score: {score:.2f}")