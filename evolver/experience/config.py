"""
Experience Manager Configuration
"""


# --- Reflection Parameters ---
# Similarity threshold to consider a new summary as matching an existing principle.
SIMILARITY_THRESHOLD = 0.85

# When a principle is associated with trajectories, we only keep the N most recent ones
MAX_SUCCESS_TRAJECTORIES_PER_PRINCIPLE = 3
MAX_FAILED_TRAJECTORIES_PER_PRINCIPLE = 2


# --- Retrieval Parameters ---
# Default number of principles to retrieve.
TOP_K_PRINCIPLES = 3

# When retrieving trajectories for a principle, how many examples to fetch.
RETRIEVE_K_SUCCESS_TRAJECTORIES = 1
RETRIEVE_K_FAILED_TRAJECTORIES = 1


# --- Trajectory Compaction Parameters ---
# To avoid exceeding LLM context windows, we truncate parts of the trajectory log
# before passing them to the agent.
MAX_DOC_CHARS = 200  # Max characters of a retrieved document to show.
MAX_THOUGHT_CHARS = 300 # Max characters of the 'thought' field to show.
