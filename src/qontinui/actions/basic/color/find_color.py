"""Find color implementation - ported from Qontinui framework.

Color-based pattern matching and scene classification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
import logging
import numpy as np
from sklearn.cluster import KMeans
import cv2

from ....object_collection import ObjectCollection
from ....action_result import ActionResult
from .color_find_options import ColorFindOptions, ColorStrategy
from ....model.match.match import Match
from ....model.element.location import Location
from ....model.element.region import Region
from ....model.element.color import RGB, HSV

logger = logging.getLogger(__name__)


@dataclass
class ColorProfile:
    """Color statistics for an image region.
    
    Used for MU strategy color matching.
    """
    h_min: float
    h_max: float
    h_mean: float
    h_std: float
    s_min: float
    s_max: float
    s_mean: float
    s_std: float
    v_min: float
    v_max: float
    v_mean: float
    v_std: float
    
    def matches(self, hsv: HSV, std_range: float = 2.0) -> bool:
        """Check if HSV color matches this profile.
        
        Args:
            hsv: Color to check
            std_range: Number of standard deviations for tolerance
            
        Returns:
            True if color matches profile
        """
        # Check hue (circular)
        h_low = (self.h_mean - self.h_std * std_range) % 360
        h_high = (self.h_mean + self.h_std * std_range) % 360
        if h_low > h_high:  # Wraps around 0
            h_matches = hsv.h >= h_low or hsv.h <= h_high
        else:
            h_matches = h_low <= hsv.h <= h_high
        
        # Check saturation
        s_low = max(0, self.s_mean - self.s_std * std_range)
        s_high = min(255, self.s_mean + self.s_std * std_range)
        s_matches = s_low <= hsv.s <= s_high
        
        # Check value
        v_low = max(0, self.v_mean - self.v_std * std_range)
        v_high = min(255, self.v_mean + self.v_std * std_range)
        v_matches = v_low <= hsv.v <= v_high
        
        return h_matches and s_matches and v_matches


@dataclass
class FindColor:
    """Color-based pattern matching implementation.
    
    Port of FindColor from Qontinui framework class.
    
    Implements color-based matching using k-means clustering,
    mean color statistics, or multi-class classification.
    
    Workflow:
    1. Acquire scenes (screenshot or provided images)
    2. Gather classification images (targets and context)
    3. Perform pixel-level color classification
    4. Extract contiguous regions as match candidates
    5. Filter and sort matches by size or score
    """
    
    # Cache for color profiles
    _profile_cache: Dict[str, ColorProfile] = field(default_factory=dict)
    
    def find(self, matches: ActionResult, 
             object_collections: List[ObjectCollection]) -> None:
        """Find matches based on color.
        
        Args:
            matches: ActionResult to populate with found matches
            object_collections: List of object collections containing targets and options
        """
        if not object_collections:
            logger.warning("No object collections provided for color finding")
            return
        
        # Get options from first collection
        first_collection = object_collections[0]
        options = first_collection.action_options.find_options.color_find_options
        
        if options.diameter < 0:
            return
        
        # Get scene to analyze
        scene = self._capture_scene(options)
        if scene is None:
            logger.warning("Could not capture scene for color finding")
            return
        
        # Perform color-based matching based on strategy
        found_matches = []
        if options.color_strategy == ColorStrategy.KMEANS:
            found_matches = self._find_kmeans(scene, first_collection, options)
        elif options.color_strategy == ColorStrategy.MU:
            found_matches = self._find_mu(scene, first_collection, options)
        else:  # CLASSIFICATION
            found_matches = self._find_classification(scene, first_collection, options)
        
        # Filter matches by area
        found_matches = self._filter_by_area(found_matches, options)
        
        # Sort matches
        if options.color_strategy == ColorStrategy.CLASSIFICATION:
            # Sort by region size (largest first)
            found_matches.sort(key=lambda m: m.region.width * m.region.height if m.region else 0, 
                        reverse=True)
        else:
            # Sort by similarity score (highest first)
            found_matches.sort(key=lambda m: m.similarity, reverse=True)
        
        # Limit number of matches
        if options.max_matches > 0:
            found_matches = found_matches[:options.max_matches]
        
        # Add matches to ActionResult
        for match in found_matches:
            matches.add_match(match)
    
    def _find_kmeans(self, scene: np.ndarray, object_collection: ObjectCollection,
                     options: ColorFindOptions) -> List[Match]:
        """Find using k-means clustering.
        
        Args:
            scene: Scene image
            object_collection: Target objects
            options: Color options
            
        Returns:
            List of matches
        """
        matches = []
        
        # Get target images
        target_images = self._get_target_images(object_collection)
        if not target_images:
            logger.warning("No target images for k-means color finding")
            return []
        
        for target in target_images:
            # Get dominant colors from target using k-means
            target_colors = self._get_kmeans_colors(target, options.kmeans)
            
            # Find regions in scene matching these colors
            for color in target_colors:
                color_matches = self._find_color_regions(
                    scene, color, options.diameter, tolerance=30
                )
                matches.extend(color_matches)
        
        return matches
    
    def _find_mu(self, scene: np.ndarray, object_collection: ObjectCollection,
                 options: ColorFindOptions) -> List[Match]:
        """Find using mean/std color statistics.
        
        Args:
            scene: Scene image
            object_collection: Target objects
            options: Color options
            
        Returns:
            List of matches
        """
        matches = []
        
        # Get target images
        target_images = self._get_target_images(object_collection)
        if not target_images:
            logger.warning("No target images for MU color finding")
            return []
        
        for target in target_images:
            # Get color profile from target
            profile = self._get_color_profile(target)
            
            # Find regions matching profile
            profile_matches = self._find_profile_regions(
                scene, profile, options.diameter
            )
            matches.extend(profile_matches)
        
        return matches
    
    def _find_classification(self, scene: np.ndarray, object_collection: ObjectCollection,
                            options: ColorFindOptions) -> List[Match]:
        """Find using multi-class classification.
        
        Args:
            scene: Scene image
            object_collection: Target and context objects
            options: Color options
            
        Returns:
            List of matches
        """
        # Get all classification images
        class_images = self._get_classification_images(object_collection)
        if not class_images:
            logger.warning("No images for classification")
            return []
        
        # Build color profiles for each class
        class_profiles = {}
        for i, img in enumerate(class_images):
            class_profiles[i] = self._get_color_profile(img)
        
        # Classify each pixel in scene
        h, w = scene.shape[:2]
        classification = np.zeros((h, w), dtype=np.int32)
        
        scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
        
        for y in range(h):
            for x in range(w):
                pixel_hsv = HSV(
                    scene_hsv[y, x, 0],
                    scene_hsv[y, x, 1],
                    scene_hsv[y, x, 2]
                )
                
                # Find best matching class
                best_class = -1
                best_score = 0.0
                
                for class_id, profile in class_profiles.items():
                    if profile.matches(pixel_hsv):
                        # Simple scoring based on distance from mean
                        score = self._calculate_profile_score(pixel_hsv, profile)
                        if score > best_score:
                            best_score = score
                            best_class = class_id
                
                classification[y, x] = best_class
        
        # Extract contiguous regions for each class
        matches = []
        for class_id in class_profiles.keys():
            if class_id == -1:
                continue
            
            # Create binary mask for this class
            mask = (classification == class_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                region = Region(x, y, w, h)
                
                match = Match(
                    location=region.center,
                    region=region,
                    similarity=0.9,  # Classification doesn't have similarity
                    pattern_id=f"class_{class_id}"
                )
                matches.append(match)
        
        return matches
    
    def _get_kmeans_colors(self, image: np.ndarray, n_clusters: int) -> List[RGB]:
        """Get dominant colors using k-means.
        
        Args:
            image: Input image
            n_clusters: Number of clusters
            
        Returns:
            List of dominant RGB colors
        """
        # Reshape image to list of pixels
        pixels = image.reshape((-1, 3))
        
        # Apply k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers as RGB colors
        colors = []
        for center in kmeans.cluster_centers_:
            rgb = RGB(int(center[2]), int(center[1]), int(center[0]))  # BGR to RGB
            colors.append(rgb)
        
        return colors
    
    def _get_color_profile(self, image: np.ndarray) -> ColorProfile:
        """Calculate color statistics for image.
        
        Args:
            image: Input image
            
        Returns:
            Color profile with statistics
        """
        # Check cache
        cache_key = str(image.shape) + str(image.sum())
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate statistics for each channel
        h_channel = hsv[:, :, 0].flatten()
        s_channel = hsv[:, :, 1].flatten()
        v_channel = hsv[:, :, 2].flatten()
        
        profile = ColorProfile(
            h_min=float(np.min(h_channel)),
            h_max=float(np.max(h_channel)),
            h_mean=float(np.mean(h_channel)),
            h_std=float(np.std(h_channel)),
            s_min=float(np.min(s_channel)),
            s_max=float(np.max(s_channel)),
            s_mean=float(np.mean(s_channel)),
            s_std=float(np.std(s_channel)),
            v_min=float(np.min(v_channel)),
            v_max=float(np.max(v_channel)),
            v_mean=float(np.mean(v_channel)),
            v_std=float(np.std(v_channel))
        )
        
        # Cache result
        self._profile_cache[cache_key] = profile
        
        return profile
    
    def _find_color_regions(self, scene: np.ndarray, color: RGB,
                           min_size: int, tolerance: int = 30) -> List[Match]:
        """Find regions matching a specific color.
        
        Args:
            scene: Scene to search
            color: Target color
            min_size: Minimum region size
            tolerance: Color tolerance
            
        Returns:
            List of matches
        """
        # Convert color to HSV for better matching
        target_hsv = color.to_hsv()
        
        # Create color range
        lower = np.array([
            max(0, target_hsv.h - tolerance),
            max(0, target_hsv.s - tolerance),
            max(0, target_hsv.v - tolerance)
        ])
        upper = np.array([
            min(179, target_hsv.h + tolerance),
            min(255, target_hsv.s + tolerance),
            min(255, target_hsv.v + tolerance)
        ])
        
        # Convert scene to HSV
        scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(scene_hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        matches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                region = Region(x, y, w, h)
                match = Match(
                    location=region.center,
                    region=region,
                    similarity=0.85
                )
                matches.append(match)
        
        return matches
    
    def _find_profile_regions(self, scene: np.ndarray, profile: ColorProfile,
                             min_size: int) -> List[Match]:
        """Find regions matching a color profile.
        
        Args:
            scene: Scene to search
            profile: Target color profile
            min_size: Minimum region size
            
        Returns:
            List of matches
        """
        h, w = scene.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
        
        # Check each pixel against profile
        for y in range(h):
            for x in range(w):
                pixel_hsv = HSV(
                    scene_hsv[y, x, 0],
                    scene_hsv[y, x, 1],
                    scene_hsv[y, x, 2]
                )
                if profile.matches(pixel_hsv):
                    mask[y, x] = 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        matches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                region = Region(x, y, w, h)
                match = Match(
                    location=region.center,
                    region=region,
                    similarity=0.8
                )
                matches.append(match)
        
        return matches
    
    def _calculate_profile_score(self, hsv: HSV, profile: ColorProfile) -> float:
        """Calculate how well a color matches a profile.
        
        Args:
            hsv: Color to score
            profile: Target profile
            
        Returns:
            Score (0.0-1.0)
        """
        # Calculate normalized distances
        h_dist = abs(hsv.h - profile.h_mean) / 180.0
        s_dist = abs(hsv.s - profile.s_mean) / 255.0
        v_dist = abs(hsv.v - profile.v_mean) / 255.0
        
        # Combined score (inverse of distance)
        avg_dist = (h_dist + s_dist + v_dist) / 3.0
        return max(0.0, 1.0 - avg_dist)
    
    def _filter_by_area(self, matches: List[Match], options: ColorFindOptions) -> List[Match]:
        """Filter matches by area constraints.
        
        Args:
            matches: Matches to filter
            options: Color options with area filtering
            
        Returns:
            Filtered matches
        """
        filtered = []
        
        for match in matches:
            if not match.region:
                continue
            
            area = match.region.width * match.region.height
            
            # Check minimum area
            if area < options.area_filtering.min_area:
                continue
            
            # Check maximum area
            if options.area_filtering.max_area > 0 and area > options.area_filtering.max_area:
                continue
            
            filtered.append(match)
        
        logger.debug(f"Filtered {len(matches) - len(filtered)} matches by area")
        return filtered
    
    def _get_target_images(self, object_collection: ObjectCollection) -> List[np.ndarray]:
        """Get target images from collection.
        
        Args:
            object_collection: Object collection
            
        Returns:
            List of target images
        """
        # This would extract actual images from StateImages in collection
        # For now, return empty list as placeholder
        return []
    
    def _get_classification_images(self, object_collection: ObjectCollection) -> List[np.ndarray]:
        """Get all images for classification.
        
        Args:
            object_collection: Object collection
            
        Returns:
            List of classification images
        """
        # This would extract all images including context
        # For now, return empty list as placeholder
        return []
    
    def _capture_scene(self, options: ColorFindOptions) -> Optional[np.ndarray]:
        """Capture scene for analysis.
        
        Args:
            options: Color options
            
        Returns:
            Scene image or None
        """
        # This would capture screenshot or use provided scene
        # For now, return None as placeholder
        logger.debug("Capturing scene for color analysis")
        return None