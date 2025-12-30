"""Coordinate system representation and operations."""

from typing import Optional
import numpy as np

from .transforms import trs2D



class System:
    """Represents a 2D coordinate system in a hierarchical transformation tree.
    
    A coordinate system defines a local frame of reference that can be nested within
    a parent system, allowing for hierarchical transformations. Each system stores
    a 3x3 affine transformation matrix (in homogeneous coordinates) that describes
    its position, rotation, and scale relative to its parent.
    
    Attributes:
        transform: 3x3 affine transformation matrix from this system to its parent.
                  Defaults to identity if not specified.
        parent: Optional parent coordinate system. If None, this is a root/global system.
    
    Examples:
        >>> # Create a root coordinate system
        >>> root = System()
        >>> 
        >>> # Create a child system translated by (5, 3) relative to root
        >>> child = System(transform=translate2D(5, 3), parent=root)
        >>> 
        >>> # Get transformation to global space
        >>> global_t = child.global_transform()
    """
    def __init__(self, transform: Optional[np.ndarray] = None, parent: Optional['System'] = None):
        """Initialize a coordinate system.
        
        Args:
            transform: 3x3 affine transformation matrix relative to parent.
                      If None, uses identity (no transformation).
            parent: Parent coordinate system. If None, this is a root system.
        """
        self.transform = transform if transform is not None else np.eye(3)
        self.parent = parent

    def global_transform(self) -> np.ndarray:
        """Computes the cumulative transformation matrix from this system to global space.
        
        Recursively multiplies transformation matrices up the hierarchy to compute
        the complete transformation from this coordinate system to the root (global)
        coordinate system.
        
        Returns:
            3x3 numpy array representing the transformation from local to global coordinates.
        
        Examples:
            >>> root = System(transform=translate2D(10, 5))
            >>> child = System(transform=translate2D(3, 2), parent=root)
            >>> global_t = child.global_transform()
            >>> # global_t represents translation by (13, 7)
        """
        if self.parent is None:
            return self.transform
        else:
            return self.parent.global_transform() @ self.transform

    def compute_convert_transform(self, target_system: 'System') -> np.ndarray:
        """Computes the transformation matrix to convert coordinates from this system to another.
        
        Calculates the transformation needed to express coordinates defined in this
        coordinate system in the target coordinate system. This is computed by:
        1. Transforming from this system to global space
        2. Transforming from global space to the target system
        
        Args:
            target_system: The destination coordinate system.
        
        Returns:
            3x3 transformation matrix that converts coordinates from this system
            to the target system.
        
        Examples:
            >>> system_a = System(transform=translate2D(5, 0))
            >>> system_b = System(transform=translate2D(0, 3))
            >>> convert_t = system_a.compute_convert_transform(system_b)
            >>> # Use convert_t to express system_a coordinates in system_b
        """
        inv_transform = np.linalg.inv(target_system.global_transform())
        return inv_transform @ self.global_transform()


def system_factory(parent: Optional[System], tx: float=0.0, ty: float=0.0, angle_rad: float=0.0, sx: float=1.0, sy: float=1.0) -> System:
    """Factory function to create a coordinate system using TRS (Translation-Rotation-Scale) parameters.
    
    Convenience function that constructs a coordinate system from intuitive transformation
    parameters instead of requiring a raw transformation matrix. The transformations are
    applied in TRS order: scale first, then rotate, then translate.
    
    Args:
        parent: Parent coordinate system. If None, creates a root system.
        tx: Translation along X-axis (default: 0.0)
        ty: Translation along Y-axis (default: 0.0)
        angle_rad: Rotation angle in radians, counter-clockwise (default: 0.0)
        sx: Scale factor along X-axis (default: 1.0)
        sy: Scale factor along Y-axis (default: 1.0)
    
    Returns:
        A new System with the specified transformation relative to its parent.
    
    Examples:
        >>> # Create root system at (10, 5) with no rotation or scaling
        >>> root = system_factory(None, tx=10, ty=5)
        >>> 
        >>> # Create child rotated 90° and scaled 2x
        >>> child = system_factory(root, angle_rad=np.pi/2, sx=2, sy=2)
    """
    transform = trs2D(tx, ty, angle_rad, sx, sy)
    return System(transform=transform, parent=parent)
