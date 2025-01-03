import numpy as np
import sys
from scipy.sparse import csr_matrix
from annoy import AnnoyIndex
from data.mock_data import get_initial_data
import os
import time

class SANM:
    def __init__(self, feature_dimension, similarity_threshold=0.7, num_trees=10, index_path="sanm_index.ann"):
        """
        Initializes the Sparse Associative Neural Memory.

        Args:
            feature_dimension (int): The number of features in the sparse representations
            similarity_threshold (float):  The threshold above which two
                                         representations are considered similar.
            num_trees (int): Number of trees to use when building the Annoy index.
            index_path (str): Path to store and load the annoy index
        """
        self.memory = []  # List of (sparse_matrix, index_in_annoy)
        self.similarity_threshold = similarity_threshold
        self.feature_dimension = feature_dimension
        self.annoy_index = AnnoyIndex(feature_dimension, 'euclidean')
        self.annoy_index_count = 0  # Keep track of the index in Annoy
        self.num_trees = num_trees
        self.index_path = index_path
        self.index_loaded = False

        #Load existing index if exists
        if os.path.exists(self.index_path):
            print("Loading Existing Index")
            try:
                self.annoy_index.load(self.index_path)
                self.index_loaded = True
                # Load existing memory from the loaded index
                for i in range(self.annoy_index.get_n_items()):
                    sparse_vector = self._array_to_sparse(self.annoy_index.get_item_vector(i))
                    self.memory.append((sparse_vector, i))
                    self.annoy_index_count +=1
            except Exception as e:
                print(f"Error loading index: {e}")
                print("Creating New Index")
                self._remove_index_with_retry(self.index_path)
        else:
            print("No Index Found, Creating New Index")
        print("SANM initialized with scipy.sparse and annoy.")

    def _remove_index_with_retry(self, index_path, max_retries=5, delay=0.1):
        """Removes the index file with retries."""
        for attempt in range(max_retries):
            try:
                os.remove(index_path)
                print(f"Index file {index_path} removed successfully.")
                return
            except PermissionError as e:
                print(f"Attempt {attempt + 1}: PermissionError removing index file, retrying in {delay} seconds...")
                time.sleep(delay)
            except FileNotFoundError:
                print(f"Index file {index_path} not found, no need to remove")
                return
            except Exception as e:
                print(f"An unexpected error occured: {e}")
                return
        print(f"Failed to remove index file {index_path} after {max_retries} retries.")


    def _build_index(self):
         """Builds the annoy index"""
         print(f"Building Annoy index with {self.num_trees} trees")
         self.annoy_index.build(self.num_trees)

    def add(self, information_chunk):
        """
        Adds a new information chunk to the memory.
        Args:
            information_chunk (dict): A dictionary representing the sparse
                                       representation of the information.
        """
        print(f"\nAttempting to add: {information_chunk}")

        sparse_vector = self._dict_to_sparse(information_chunk)
        
        if not self.memory:
          self.memory.append((sparse_vector, self.annoy_index_count))
          if not self.index_loaded:
            self.annoy_index.add_item(self.annoy_index_count, self._normalize_vector(sparse_vector).toarray().flatten())
          self.annoy_index_count += 1
          print(f"  Added as the first chunk.")
          return

        best_match_index = -1
        max_similarity = 0
        
        #Find the best_match using cosine similarity
        for i, (mem_sparse, mem_annoy) in enumerate(self.memory):
            similarity = self._calculate_similarity(sparse_vector, mem_sparse)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_index = i
        if best_match_index != -1:
            print(f"Best Similarity Match {best_match_index}: Similarity: {max_similarity:.2f}")
        else:
            print("No similarity match found")


        if max_similarity >= self.similarity_threshold:
            print(f"  Merging with chunk {best_match_index} (similarity: {max_similarity:.2f})")
            self.memory[best_match_index] = (self._merge_chunks(self.memory[best_match_index][0], sparse_vector), self.memory[best_match_index][1])
            if not self.index_loaded:
                self.annoy_index.add_item(self.memory[best_match_index][1], self._normalize_vector(self.memory[best_match_index][0]).toarray().flatten())
        else:
            self.memory.append((sparse_vector, self.annoy_index_count))
            if not self.index_loaded:
                self.annoy_index.add_item(self.annoy_index_count, self._normalize_vector(sparse_vector).toarray().flatten())
            self.annoy_index_count += 1
            print(f"  Added as a new chunk.")

        print(f"Current memory size: {len(self.memory)}")
        sys.stdout.flush()
    
    def save_index(self):
        """Saves the current Annoy Index"""
        if not self.index_loaded:
            self._build_index()
            self.annoy_index.save(self.index_path)
            print(f"Annoy index saved to {self.index_path}")
        else:
            print(f"Index is already loaded and cannot be saved.")

    def query(self, query_chunk):
        """
        Queries the memory for information similar to the query chunk.
        Args:
            query_chunk (dict): The sparse representation of the query.
        Returns:
            list: A list of tuples, where each tuple contains (memory_chunk, similarity_score)
                  for chunks that are similar to the query.
        """
        print(f"\nQuerying memory with: {query_chunk}")
        sparse_query = self._dict_to_sparse(query_chunk)

        # Find nearest neighbors using Annoy
        nearest_neighbors = self.annoy_index.get_nns_by_vector(self._normalize_vector(sparse_query).toarray().flatten(), 10, include_distances=True)  # Get top 10 results
        results = []
        
        if len(nearest_neighbors[0]) > 0:
            for i, mem_index in enumerate(nearest_neighbors[0]):
                for index, (mem_sparse, mem_annoy) in enumerate(self.memory):
                    if mem_index == mem_annoy:
                        similarity = self._calculate_similarity(sparse_query, mem_sparse)
                        if similarity >= self.similarity_threshold:
                          results.append((mem_sparse, similarity))
                          print(f"  Found match in chunk {index}: Similarity = {similarity:.2f}")
                        else:
                            print(f"  No match in chunk {index}: Similarity = {similarity:.2f}")


        sorted_results = sorted(results, key=lambda item: item[1], reverse=True) # Sort by similarity
        print("\nQuery Results:")
        if sorted_results:
            for chunk, similarity in sorted_results:
                print(f"  Chunk: {self._sparse_to_dict(chunk)}, Similarity: {similarity:.2f}")
        else:
            print("  No matching chunks found.")
        sys.stdout.flush()
        return sorted_results

    def _dict_to_sparse(self, information_chunk):
        """Converts a dictionary to a sparse matrix"""
        indices = []
        data = []
        for feature, value in information_chunk.items():
            indices.append(int(feature))
            data.append(value)
        sparse_vector = csr_matrix((data, ([0] * len(data), indices)), shape=(1, self.feature_dimension))
        #print(f"Sparse Vector Before Add: {sparse_vector.toarray().flatten()}, Type: {type(sparse_vector.toarray().flatten())}")
        return sparse_vector
    
    def _array_to_sparse(self, array):
        """Converts an array to a sparse matrix"""
        data = []
        indices = []
        for index, value in enumerate(array):
            if value != 0:
                data.append(value)
                indices.append(index)
        
        return csr_matrix((data, ([0] * len(data), indices)), shape=(1, self.feature_dimension))
    
    def _normalize_vector(self, sparse_vector):
        """Normalizes the sparse vector"""
        norm = np.sqrt(sparse_vector.dot(sparse_vector.T).toarray()[0][0])
        if norm == 0:
           return sparse_vector
        return sparse_vector/norm

    def _sparse_to_dict(self, sparse_matrix):
        """Converts a sparse matrix back to a dictionary for printing."""
        
        sparse_matrix = sparse_matrix.tocoo()
        result = {}
        for i, j, v in zip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):
            result[str(j)] = v
        return result

    def _calculate_similarity(self, chunk1, chunk2):
        """Calculates the cosine similarity between two sparse matrices."""
        dot_product = chunk1.dot(chunk2.T).toarray()[0][0]
        magnitude1 = np.sqrt(chunk1.dot(chunk1.T).toarray()[0][0])
        magnitude2 = np.sqrt(chunk2.dot(chunk2.T).toarray()[0][0])

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        similarity = dot_product / (magnitude1 * magnitude2)
        #print(f"Similarity of {similarity:.2f} for vectors {chunk1.toarray().flatten()} and {chunk2.toarray().flatten()}")
        return similarity

    def _merge_chunks(self, chunk1, chunk2):
        """Merges two similar chunks (averaging shared features)."""
        merged_chunk = chunk1 + chunk2
        return merged_chunk

# Example Usage
if __name__ == "__main__":
    # Load initial data
    initial_data = get_initial_data()

    # Determine the feature dimension (max feature index + 1)
    feature_dimension = 0
    for data_chunk in initial_data:
      for feature in data_chunk:
          feature_dimension = max(feature_dimension, int(feature))
    feature_dimension += 1

    sanm = SANM(feature_dimension=feature_dimension, similarity_threshold=0.6, num_trees=10)
    print("\nLoading initial data:")

    for data_chunk in initial_data:
        sanm.add(data_chunk)
    
    sanm.save_index()

    # Query the memory
    query = {"1": 0.9, "3": 0.7}
    sanm.query(query)

    query_2 = {"2": 0.8, "5": 0.95}
    sanm.query(query_2)