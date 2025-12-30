"""Coordinate representation classes for points and vectors."""

from typing import Optional
import numpy as np

from .system import System
from .types import CoordinateType


def transform_coordinate(transform: np.ndarray, coordinates: np.ndarray, coordinate_type: CoordinateType) -> np.ndarray:
    """Applies an affine transformation to a coordinate, respecting point vs vector semantics.
    
    Points and vectors transform differently under affine transformations:
    - Points (weight=1): Affected by translation, rotation, and scaling
    - Vectors (weight=0): Affected only by rotation and scaling, NOT translation
    
    This function converts to homogeneous coordinates, applies the transformation,
    and converts back to Cartesian coordinates.
    
    Args:
        transform: 3x3 affine transformation matrix in homogeneous coordinates.
        coordinates: 2D coordinate as numpy array [x, y].
        coordinate_type: CoordinateType.POINT or CoordinateType.VECTOR.
    
    Returns:
        Transformed 2D coordinate as numpy array [x', y'].
    
    Examples:
        >>> # Point translation
        >>> transform_coordinate(translate2D(5, 3), np.array([1, 2]), CoordinateType.POINT)
        array([6., 5.])  # Point moved by (5, 3)
        
        >>> # Vector translation (no effect)
        >>> transform_coordinate(translate2D(5, 3), np.array([1, 2]), CoordinateType.VECTOR)
        array([1., 2.])  # Vector unchanged
    """
    # Convert to homogeneous coordinates
    weight = 1.0 if coordinate_type == CoordinateType.POINT else 0.0
    homogeneous_point = np.append(coordinates, weight) 
    transformed_point = transform @ homogeneous_point

    # Return to Cartesian coordinates
    # Normalize if necessary
    weight = transformed_point[2]
    if weight != 0:
        transformed_point /= weight
    return transformed_point[:2]


class Coordinate:
    """Base class for representing coordinates (points or vectors) in a coordinate system.
    
    Coordinates can be defined in any coordinate system and converted between systems.
    The distinction between points and vectors is crucial:
    - Points: Represent positions, affected by all transformations including translation
    - Vectors: Represent directions/displacements, unaffected by translation
    
    Attributes:
        coordinate_type: CoordinateType.POINT or CoordinateType.VECTOR
        local_coords: 2D numpy array [x, y] in the local coordinate system
        system: The coordinate system this coordinate is defined in
    
    Examples:
        >>> system = System(transform=translate2D(5, 3))
        >>> coord = Coordinate(CoordinateType.POINT, np.array([1, 2]), system)
        >>> global_coord = coord.to_global()
    """

    def __init__(self, coordinate_type: CoordinateType, local_coords: np.ndarray, system: Optional[System] = None):
        """Initialize a coordinate.
        
        Args:
            coordinate_type: CoordinateType.POINT or CoordinateType.VECTOR
            local_coords: 2D numpy array [x, y] in the local coordinate system
            system: Coordinate system this coordinate is defined in.
                   If None, uses global/identity system.
        """
        self.coordinate_type = coordinate_type
        self.local_coords = local_coords
        self.system = system if system is not None else System()

    def to_global(self) -> 'Coordinate':
        """Converts this coordinate to global (identity) coordinate system.
        
        Applies the cumulative transformation from this coordinate's system through
        all parent systems to express the coordinate in global space.
        
        Returns:
            New Coordinate with coordinates expressed in global system.
        
        Examples:
            >>> root = System(transform=translate2D(10, 5))
            >>> child = System(transform=translate2D(3, 2), parent=root)
            >>> point = Point(np.array([1, 1]), system=child)
            >>> global_point = point.to_global()
            >>> global_point.local_coords  # Should be [14, 8]
        """
        global_transform = self.system.global_transform()
        global_coords = transform_coordinate(global_transform, self.local_coords, self.coordinate_type)
        return Coordinate(local_coords=global_coords, coordinate_type=self.coordinate_type, system=None)
        
    def to_system(self, target_system: System) -> 'Coordinate':
        """Converts this coordinate to a different coordinate system.
        
        Transforms the coordinate from its current system to the target system,
        properly handling the coordinate type (point vs vector) semantics.
        
        Args:
            target_system: The destination coordinate system.
        
        Returns:
            New Coordinate with coordinates expressed in the target system.
        
        Examples:
            >>> system_a = System(transform=translate2D(5, 0))
            >>> system_b = System(transform=translate2D(0, 3))
            >>> point_in_a = Point(np.array([0, 0]), system=system_a)
            >>> point_in_b = point_in_a.to_system(system_b)
            >>> point_in_b.local_coords  # Should be [5, -3]
        """
        # Inverse transform from global to target system
        convert_transform = self.system.compute_convert_transform(target_system)
        new_local_coords = transform_coordinate(convert_transform, self.local_coords, self.coordinate_type)
        return Coordinate(local_coords=new_local_coords, coordinate_type=self.coordinate_type, system=target_system)


class Point(Coordinate):
    """Represents a point (position) in a coordinate system.
    
    Points are affected by all transformations including translation, rotation, and scaling.
    Use this class to represent positions in space.
    
    Args:
        local_coords: 2D numpy array [x, y] representing the point position
        system: Coordinate system this point is defined in. If None, uses global system.
    
    Examples:
        >>> # Point at origin in a translated system
        >>> system = System(transform=translate2D(10, 5))
        >>> point = Point(np.array([0, 0]), system=system)
        >>> global_point = point.to_global()
        >>> global_point.local_coords  # [10, 5] - affected by translation
    """
    
    def __init__(self, local_coords: np.ndarray, system: Optional[System] = None):
        super().__init__(
            coordinate_type=CoordinateType.POINT,
            local_coords=local_coords, 
            system=system)


class Vector(Coordinate):
    """Represents a vector (direction/displacement) in a coordinate system.
    
    Vectors are NOT affected by translation, only by rotation and scaling.
    Use this class to represent directions, velocities, or relative displacements.
    
    Args:
        local_coords: 2D numpy array [x, y] representing the vector components
        system: Coordinate system this vector is defined in. If None, uses global system.
    
    Examples:
        >>> # Vector in a translated system
        >>> system = System(transform=translate2D(10, 5))
        >>> vector = Vector(np.array([1, 0]), system=system)
        >>> global_vector = vector.to_global()
        >>> global_vector.local_coords  # Still [1, 0] - unaffected by translation
    """
    
    def __init__(self, local_coords: np.ndarray, system: Optional[System] = None):
        super().__init__(
            coordinate_type=CoordinateType.VECTOR,
            local_coords=local_coords, 
            system=system)
