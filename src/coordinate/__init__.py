import dataclasses
import numpy as np

def translate2D(tx: float, ty: float) -> np.ndarray:
    """Creates a 2D translation matrix."""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

def rotate2D(angle_rad: float) -> np.ndarray:
    """Creates a 2D rotation matrix."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def scale2D(sx: float, sy: float) -> np.ndarray:
    """Creates a 2D scaling matrix."""
    return np.array([[sx, 0,  0],
                     [0, sy,  0],
                     [0,  0, 1]])

def trs2D(tx: float, ty: float, angle_rad: float, sx: float, sy: float) -> np.ndarray:
    """Creates a combined translation, rotation, and scaling matrix."""
    T = translate2D(tx, ty)
    R = rotate2D(angle_rad)
    S = scale2D(sx, sy)
    return T @ R @ S


class CoordinateType:
    POINT: str = "point"
    VECTOR: str = "vector"

def transform_coordinate(transform: np.ndarray, coordinates: np.ndarray, coordinate_type: CoordinateType) -> np.ndarray:
    """Applies an affine transformation to a coordinate point or vector."""

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


@dataclasses.dataclass
class System:
    transform: np.ndarray  # Local to Parent, affine 3x3 matrix
    parent: 'System' | None

    def global_transform(self) -> np.ndarray:
        if self.parent is None:
            return self.transform
        else:
            return self.parent.global_transform() @ self.transform
        
    def compute_convert_transform(self, target_system: 'System') -> np.ndarray:
        inv_transform = np.linalg.inv(target_system.global_transform())
        return inv_transform @ self.system.global_transform()
    
def system_factory(parent: 'System' | None, tx: float, ty: float, angle_rad: float, sx: float, sy: float) -> System:
    transform = trs2D(tx, ty, angle_rad, sx, sy)
    return System(transform=transform, parent=parent)


class Coordinate:
    def __init__(self, coordinate_type: CoordinateType, local_coords: np.ndarray, system: System | None=None):
        self.coordinate_type = coordinate_type
        self.local_coords = local_coords
        if self.system is None:
            identity_system = System(transform=np.eye(3), parent=None)
            self.system = identity_system
        self.system = system

    def to_global(self) -> 'Coordinate':
        if self.system.parent is None:
            return self
        else:
            parent_point = Coordinate(
                system=self.system.parent,
                local_coords=self.system.transform @ np.append(self.local_coords, 1)
            )
            return parent_point.to_global()
        
    def to_system(self, target_system: System) -> 'Point':
        # Inverse transform from global to target system
        convert_transform = self.system.compute_convert_transform(target_system)
        new_local_coords = transform_coordinate(convert_transform, self.local_coords, self.coordinate_type)
        return Coordinate(system=target_system, local_coords=new_local_coords, coordinate_type=self.coordinate_type)


class Point(Coordinate):
    def __init__(self, local_coords: np.ndarray, system: System | None=None):
        super().__init__(
            coordinate_type=CoordinateType.POINT,
            local_coords=local_coords, 
            system=system)


class Vector(Coordinate):
    def __init__(self, local_coords: np.ndarray, system: System | None=None):
        super().__init__(
            coordinate_type=CoordinateType.VECTOR,
            local_coords=local_coords, 
            system=system)

