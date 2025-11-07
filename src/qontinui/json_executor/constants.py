"""Constants for JSON executor module."""

# Default similarity threshold for image matching (library default)
#
# This is the LOWEST priority in the similarity cascade:
# 1. FindOptions.similarity (action-level, explicit) - HIGHEST
# 2. Pattern.similarity (image-level)
# 3. StateImage.threshold (state image-level from JSON config)
# 4. QontinuiSettings.similarity_threshold (project config) = 0.85
# 5. DEFAULT_SIMILARITY_THRESHOLD (library default) = 0.7 - LOWEST
#
# NOTE: This should match FindActionDefaults.default_similarity_threshold
DEFAULT_SIMILARITY_THRESHOLD = 0.7
