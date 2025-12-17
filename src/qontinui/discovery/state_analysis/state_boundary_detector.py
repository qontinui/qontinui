"""State Boundary Detection Service for qontinui-runner.

This service identifies unique screen states from video frames using visual clustering
and analysis techniques. It combines multiple approaches:
- SSIM (Structural Similarity Index) for frame comparison
- Perceptual hashing for fast initial grouping
- Feature extraction (ORB, SIFT) for robust matching
- Optical flow analysis for detecting transitions
- Multiple clustering algorithms (DBSCAN, hierarchical, k-means)

The service correlates visual changes with input events to identify state boundaries
and transitions, enabling automated UI state mapping for testing and automation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

# Import models from the models package
from qontinui.discovery.state_analysis.models import DetectedState, Frame, InputEvent

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class StateBoundaryConfig:
    """Configuration for state boundary detection.

    Attributes:
        similarity_threshold: SSIM threshold for frame similarity (0.0-1.0)
        clustering_algorithm: Algorithm to use ("dbscan", "hierarchical", "kmeans")
        min_state_duration_ms: Minimum duration for a state to be valid
        feature_extractor: Feature extraction method ("orb", "sift", "surf")
        feature_count: Number of features to extract per frame
        optical_flow_threshold: Threshold for optical flow magnitude
        phash_size: Size of perceptual hash (8 = 8x8 = 64 bits)
        phash_difference_threshold: Max hamming distance for pHash similarity
        min_frames_per_state: Minimum frames required to form a state
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN minimum samples parameter
        hierarchical_distance_threshold: Distance threshold for hierarchical clustering
        kmeans_n_clusters: Number of clusters for k-means (None = auto-detect)
    """

    similarity_threshold: float = 0.92
    clustering_algorithm: str = "dbscan"
    min_state_duration_ms: int = 500
    feature_extractor: str = "orb"
    feature_count: int = 500
    optical_flow_threshold: float = 0.3
    phash_size: int = 8
    phash_difference_threshold: int = 10
    min_frames_per_state: int = 3
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3
    hierarchical_distance_threshold: float = 1.5
    kmeans_n_clusters: int | None = None


# ============================================================================
# Helper Classes
# ============================================================================


@dataclass
class FrameFeatures:
    """Features extracted from a frame for clustering.

    Attributes:
        frame_index: Index of the frame
        timestamp_ms: Timestamp in milliseconds
        phash: Perceptual hash string
        ssim_cache: Cache of SSIM values with other frames
        keypoints: Feature keypoints (ORB, SIFT, etc.)
        descriptors: Feature descriptors
        histogram: Color histogram
        edges: Edge detection features
        optical_flow_prev: Optical flow magnitude from previous frame
    """

    frame_index: int
    timestamp_ms: int
    phash: str
    ssim_cache: dict[int, float] = field(default_factory=dict)
    keypoints: list | None = None
    descriptors: np.ndarray | None = None
    histogram: np.ndarray | None = None
    edges: np.ndarray | None = None
    optical_flow_prev: float = 0.0


@dataclass
class TransitionPoint:
    """Represents a transition between states.

    Attributes:
        transition_id: Unique identifier
        from_state_id: Source state ID
        to_state_id: Destination state ID
        frame_index: Frame where transition occurs
        timestamp_ms: Timestamp of transition
        trigger_event_index: Index of triggering input event
        optical_flow_magnitude: Magnitude of optical flow
        visual_change_score: Visual difference score
        transition_duration_ms: Duration of transition
        metadata: Additional metadata
    """

    transition_id: str
    from_state_id: str
    to_state_id: str
    frame_index: int
    timestamp_ms: int
    trigger_event_index: int | None = None
    optical_flow_magnitude: float = 0.0
    visual_change_score: float = 0.0
    transition_duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# State Boundary Detector
# ============================================================================


class StateBoundaryDetector:
    """Detects state boundaries in video frames using visual clustering.

    This class provides comprehensive state detection capabilities by combining
    multiple computer vision techniques to identify unique screen states and
    transitions between them.
    """

    def __init__(self, config: StateBoundaryConfig | None = None):
        """Initialize the state boundary detector.

        Args:
            config: Configuration object, uses defaults if not provided
        """
        self.config = config or StateBoundaryConfig()
        self.feature_detector = None
        self._init_feature_detector()
        logger.info(f"Initialized StateBoundaryDetector with {self.config.feature_extractor}")

    def _init_feature_detector(self):
        """Initialize the feature detector based on configuration."""
        try:
            if self.config.feature_extractor.lower() == "orb":
                self.feature_detector = cv2.ORB_create(nfeatures=self.config.feature_count)
            elif self.config.feature_extractor.lower() == "sift":
                self.feature_detector = cv2.SIFT_create(nfeatures=self.config.feature_count)
            elif self.config.feature_extractor.lower() == "surf":
                # SURF is in opencv-contrib, may not be available
                self.feature_detector = cv2.xfeatures2d.SURF_create()
            else:
                logger.warning(
                    f"Unknown feature extractor: {self.config.feature_extractor}, "
                    "defaulting to ORB"
                )
                self.feature_detector = cv2.ORB_create(nfeatures=self.config.feature_count)
        except Exception as e:
            logger.error(f"Failed to initialize feature detector: {e}, defaulting to ORB")
            self.feature_detector = cv2.ORB_create(nfeatures=self.config.feature_count)

    def detect_states(self, frames: list[Frame]) -> list[DetectedState]:
        """Detect unique states from a sequence of frames.

        This is the main entry point for state detection. The process:
        1. Extract features from each frame
        2. Compute pairwise similarity matrix
        3. Cluster frames into states
        4. Identify state transition points
        5. Return unique states with representative frames

        Args:
            frames: List of Frame objects to analyze

        Returns:
            List of DetectedState objects representing unique states

        Raises:
            ValueError: If frames list is empty or invalid
        """
        if not frames:
            raise ValueError("Cannot detect states from empty frame list")

        logger.info(f"Starting state detection for {len(frames)} frames")

        # Step 1: Extract features from all frames
        logger.info("Extracting features from frames...")
        frame_features = self._extract_all_features(frames)

        # Step 2: Compute similarity matrix
        logger.info("Computing pairwise similarity matrix...")
        similarity_matrix = self._compute_similarity_matrix(frames, frame_features)

        # Step 3: Cluster frames into states
        logger.info(f"Clustering frames using {self.config.clustering_algorithm}...")
        cluster_labels = self._cluster_frames(similarity_matrix, frame_features)

        # Step 4: Build DetectedState objects from clusters
        logger.info("Building state objects from clusters...")
        states = self._build_states_from_clusters(frames, frame_features, cluster_labels)

        # Step 5: Filter states by minimum duration
        states = self._filter_states_by_duration(states)

        logger.info(f"Detected {len(states)} unique states")
        return states

    def identify_transitions(
        self, frames: list[Frame], events: list[InputEvent]
    ) -> list[TransitionPoint]:
        """Identify state transitions and correlate with input events.

        This method detects where visual changes occur in the frame sequence
        and attempts to correlate them with user input events to identify
        which actions trigger which transitions.

        Args:
            frames: List of Frame objects
            events: List of InputEvent objects

        Returns:
            List of TransitionPoint objects
        """
        if not frames:
            return []

        logger.info(f"Identifying transitions in {len(frames)} frames with {len(events)} events")

        transitions = []
        prev_frame = None

        for i, frame in enumerate(frames):
            if prev_frame is None:
                prev_frame = frame
                continue

            # Compute visual change between consecutive frames
            visual_change = self._compute_visual_change(prev_frame, frame)
            optical_flow_mag = self._compute_optical_flow(prev_frame.image, frame.image)

            # Check if this is a significant transition
            if (
                visual_change < (1.0 - self.config.similarity_threshold)
                or optical_flow_mag > self.config.optical_flow_threshold
            ):
                # Find closest input event
                trigger_event_idx = self._find_closest_event(frame.timestamp, events)

                transition = TransitionPoint(
                    transition_id=str(uuid.uuid4()),
                    from_state_id="",  # Will be filled by state detection
                    to_state_id="",  # Will be filled by state detection
                    frame_index=i,
                    timestamp_ms=int(frame.timestamp * 1000),
                    trigger_event_index=trigger_event_idx,
                    optical_flow_magnitude=optical_flow_mag,
                    visual_change_score=visual_change,
                )
                transitions.append(transition)

            prev_frame = frame

        logger.info(f"Identified {len(transitions)} transitions")
        return transitions

    def compute_similarity(self, frame1: Frame, frame2: Frame) -> float:
        """Compute SSIM similarity between two frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            SSIM similarity score (0.0-1.0, higher is more similar)
        """
        # Convert to grayscale for SSIM computation
        gray1 = cv2.cvtColor(frame1.image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2.image, cv2.COLOR_BGR2GRAY)

        # Ensure same dimensions
        if gray1.shape != gray2.shape:
            h = min(gray1.shape[0], gray2.shape[0])
            w = min(gray1.shape[1], gray2.shape[1])
            gray1 = gray1[:h, :w]
            gray2 = gray2[:h, :w]

        # Compute SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return float(score)

    def compute_perceptual_hash(self, frame: Frame) -> str:
        """Compute perceptual hash for fast frame comparison.

        Args:
            frame: Frame to hash

        Returns:
            Perceptual hash as a string
        """
        # Convert to PIL Image
        rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Compute perceptual hash
        phash = imagehash.phash(pil_image, hash_size=self.config.phash_size)
        return str(phash)

    # ========================================================================
    # Private Methods - Feature Extraction
    # ========================================================================

    def _extract_all_features(self, frames: list[Frame]) -> list[FrameFeatures]:
        """Extract features from all frames.

        Args:
            frames: List of frames

        Returns:
            List of FrameFeatures objects
        """
        features_list = []

        prev_gray = None
        for i, frame in enumerate(frames):
            try:
                # Compute perceptual hash
                phash = self.compute_perceptual_hash(frame)

                # Extract keypoints and descriptors
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)  # type: ignore[attr-defined]

                # Compute color histogram
                histogram = self._compute_histogram(frame.image)

                # Compute edge features
                edges = cv2.Canny(gray, 100, 200)

                # Compute optical flow from previous frame
                optical_flow_mag = 0.0
                if prev_gray is not None:
                    optical_flow_mag = self._compute_optical_flow(prev_gray, gray)

                features = FrameFeatures(
                    frame_index=i,
                    timestamp_ms=int(frame.timestamp * 1000),
                    phash=phash,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    histogram=histogram,
                    edges=edges,
                    optical_flow_prev=optical_flow_mag,
                )
                features_list.append(features)
                prev_gray = gray

            except Exception as e:
                logger.error(f"Failed to extract features from frame {i}: {e}")
                # Create minimal features object
                features = FrameFeatures(
                    frame_index=i,
                    timestamp_ms=int(frame.timestamp * 1000),
                    phash="",
                )
                features_list.append(features)

        return features_list

    def _compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute color histogram for an image.

        Args:
            image: BGR image

        Returns:
            Flattened histogram array
        """
        # Compute histogram for each channel
        hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])

        # Normalize and concatenate
        hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compute_optical_flow(self, gray1: np.ndarray, gray2: np.ndarray) -> float:
        """Compute optical flow magnitude between two grayscale frames.

        Args:
            gray1: First grayscale frame
            gray2: Second grayscale frame

        Returns:
            Average magnitude of optical flow
        """
        try:
            # Ensure same size
            if gray1.shape != gray2.shape:
                h = min(gray1.shape[0], gray2.shape[0])
                w = min(gray1.shape[1], gray2.shape[1])
                gray1 = gray1[:h, :w]
                gray2 = gray2[:h, :w]

            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore[call-overload]

            # Calculate magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)

            return float(avg_magnitude)

        except Exception as e:
            logger.error(f"Failed to compute optical flow: {e}")
            return 0.0

    # ========================================================================
    # Private Methods - Similarity Computation
    # ========================================================================

    def _compute_similarity_matrix(
        self, frames: list[Frame], features_list: list[FrameFeatures]
    ) -> np.ndarray:
        """Compute pairwise similarity matrix for all frames.

        Args:
            frames: List of frames
            features_list: List of extracted features

        Returns:
            NxN similarity matrix where N is number of frames
        """
        n = len(frames)
        similarity_matrix = np.zeros((n, n))

        # Use perceptual hash for fast initial filtering
        for i in range(n):
            similarity_matrix[i, i] = 1.0  # Perfect similarity with self

            for j in range(i + 1, n):
                # Check pHash similarity first (fast)
                phash_i = imagehash.hex_to_hash(features_list[i].phash)
                phash_j = imagehash.hex_to_hash(features_list[j].phash)
                phash_distance = phash_i - phash_j

                if phash_distance <= self.config.phash_difference_threshold:
                    # Similar pHashes, compute detailed SSIM
                    similarity = self.compute_similarity(frames[i], frames[j])
                else:
                    # Very different pHashes, skip SSIM
                    similarity = 0.0

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

                # Cache SSIM result
                features_list[i].ssim_cache[j] = similarity
                features_list[j].ssim_cache[i] = similarity

        return similarity_matrix

    def _compute_visual_change(self, frame1: Frame, frame2: Frame) -> float:
        """Compute visual change score between two frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Visual change score (0.0 = identical, 1.0 = completely different)
        """
        similarity = self.compute_similarity(frame1, frame2)
        return 1.0 - similarity

    # ========================================================================
    # Private Methods - Clustering
    # ========================================================================

    def _cluster_frames(
        self, similarity_matrix: np.ndarray, features_list: list[FrameFeatures]
    ) -> np.ndarray:
        """Cluster frames based on similarity matrix.

        Args:
            similarity_matrix: Pairwise similarity matrix
            features_list: List of frame features

        Returns:
            Array of cluster labels for each frame
        """
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        if self.config.clustering_algorithm == "dbscan":
            return self._cluster_dbscan(distance_matrix)
        elif self.config.clustering_algorithm == "hierarchical":
            return self._cluster_hierarchical(distance_matrix)
        elif self.config.clustering_algorithm == "kmeans":
            return self._cluster_kmeans(distance_matrix)
        else:
            logger.warning(
                f"Unknown clustering algorithm: {self.config.clustering_algorithm}, " "using DBSCAN"
            )
            return self._cluster_dbscan(distance_matrix)

    def _cluster_dbscan(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN algorithm.

        Args:
            distance_matrix: Pairwise distance matrix

        Returns:
            Array of cluster labels
        """
        clustering = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            metric="precomputed",
        )
        labels = clustering.fit_predict(distance_matrix)
        logger.info(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels  # type: ignore[no-any-return]

    def _cluster_hierarchical(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using hierarchical clustering.

        Args:
            distance_matrix: Pairwise distance matrix

        Returns:
            Array of cluster labels
        """
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.config.hierarchical_distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)
        logger.info(f"Hierarchical clustering found {len(set(labels))} clusters")
        return labels  # type: ignore[no-any-return]

    def _cluster_kmeans(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using k-means algorithm.

        Args:
            distance_matrix: Pairwise distance matrix

        Returns:
            Array of cluster labels
        """
        # K-means needs feature vectors, use distance matrix as features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(distance_matrix)

        # Determine number of clusters
        n_clusters = self.config.kmeans_n_clusters
        if n_clusters is None:
            # Auto-detect using elbow method (simplified)
            n_clusters = min(10, max(2, len(distance_matrix) // 20))

        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(features_scaled)
        logger.info(f"K-means found {len(set(labels))} clusters")
        return labels  # type: ignore[no-any-return]

    # ========================================================================
    # Private Methods - State Building
    # ========================================================================

    def _build_states_from_clusters(
        self,
        frames: list[Frame],
        features_list: list[FrameFeatures],
        labels: np.ndarray,
    ) -> list[DetectedState]:
        """Build DetectedState objects from cluster labels.

        Args:
            frames: List of frames
            features_list: List of frame features
            labels: Cluster labels for each frame

        Returns:
            List of DetectedState objects
        """
        states = []
        unique_labels = set(labels)

        # Remove noise label (-1) if present (from DBSCAN)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            # Get all frames in this cluster
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]

            if len(cluster_indices) < self.config.min_frames_per_state:
                continue

            # Get frame indices and timestamps
            frame_indices = [features_list[i].frame_index for i in cluster_indices]
            timestamps = [features_list[i].timestamp_ms for i in cluster_indices]

            # Find representative frame (frame closest to cluster center)
            representative_idx = self._find_representative_frame(cluster_indices, frames)

            # Create state with unique ID
            state_id = f"state_{label}"
            state = DetectedState(
                id=state_id,
                name=f"State_{label}",
                description=f"Detected state with {len(cluster_indices)} frames",
                state_images=[],  # Will be populated by image extraction
                start_frame_index=min(frame_indices),
                end_frame_index=max(frame_indices),
                frame_indices=frame_indices,
                boundary=None,  # Could be computed from frame dimensions
                metadata={
                    "cluster_label": int(label),
                    "num_frames": len(cluster_indices),
                    "representative_frame_index": representative_idx,
                    "first_occurrence_ms": min(timestamps),
                    "last_occurrence_ms": max(timestamps),
                    "duration_ms": max(timestamps) - min(timestamps),
                },
            )
            states.append(state)

        return states

    def _find_representative_frame(self, cluster_indices: list[int], frames: list[Frame]) -> int:
        """Find the most representative frame in a cluster.

        This finds the frame that is most similar to all other frames in the cluster,
        i.e., the frame closest to the cluster centroid.

        Args:
            cluster_indices: Indices of frames in this cluster
            frames: List of all frames

        Returns:
            Index of the representative frame
        """
        if len(cluster_indices) == 1:
            return cluster_indices[0]

        # Compute average similarity to all other frames in cluster
        best_idx = cluster_indices[0]
        best_avg_similarity = 0.0

        for idx in cluster_indices:
            similarities = []
            for other_idx in cluster_indices:
                if idx != other_idx:
                    sim = self.compute_similarity(frames[idx], frames[other_idx])
                    similarities.append(sim)

            avg_similarity = np.mean(similarities) if similarities else 0.0
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity  # type: ignore[assignment]
                best_idx = idx

        return best_idx

    def _filter_states_by_duration(self, states: list[DetectedState]) -> list[DetectedState]:
        """Filter out states that are too short.

        Args:
            states: List of detected states

        Returns:
            Filtered list of states
        """
        filtered = []
        for state in states:
            duration = state.metadata.get("duration_ms", 0)
            if duration >= self.config.min_state_duration_ms:
                filtered.append(state)
            else:
                logger.debug(
                    f"Filtering out {state.name} with duration {duration}ms "
                    f"(min: {self.config.min_state_duration_ms}ms)"
                )

        logger.info(
            f"Filtered {len(states) - len(filtered)} states below "
            f"minimum duration ({self.config.min_state_duration_ms}ms)"
        )
        return filtered

    # ========================================================================
    # Private Methods - Utilities
    # ========================================================================

    def _find_closest_event(self, frame_timestamp: float, events: list[InputEvent]) -> int | None:
        """Find the input event closest to a frame timestamp.

        Args:
            frame_timestamp: Frame timestamp in seconds
            events: List of input events

        Returns:
            Index of closest event, or None if no events
        """
        if not events:
            return None

        closest_idx = 0
        min_diff = abs(events[0].timestamp - frame_timestamp)

        for i, event in enumerate(events):
            diff = abs(event.timestamp - frame_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        # Only return if within reasonable time window (1 second)
        if min_diff < 1.0:
            return closest_idx

        return None
