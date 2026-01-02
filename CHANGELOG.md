# Changelog

## [Unreleased]
### Added
- `Coordinate` support for ArrayLike inputs (lists, tuples) for `coords` argument in addition to numpy arrays
- `Coordinate` support for DxN arrays inputs for `coords` argument where D is dimension and N is number of coordinates, allowing batch operations
- Support for arithemitc operators, the coordinate acting as a numpy array, `__array__`, `__getitem__`, `__setitem__`, `__len__`, `__repr__`, `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__mul__`, `__rmul__`, `__truediv__`, `__rtruediv__`, `__neg__`, `__abs__`, `__eq__`, `__ne__`

### Changed
- Renamed `CoordinateType` to `CoordinateKind`
- Coordinate:
  - rename `coordinate_type` to `kind`, to avoid confusion with Python types and numpy dtypes
  - Renamed `local_coords` to `coords`, because `point.to_absolute().local_coords` was confusing
- Frame:
  - `compute_absolute_transform` now returns identity transform for root frames instead of raising
  - `create_frame` `parent` argument can now be `None` for root frames

## [0.2.0] - 2025-12-31

### Added
- Rename package to Coordinatus
- Increase test coverage for visualization module
- Publish test coverage reports to Github Pages

## [0.1.0] - 2025-12-31

### Added
- Initial release
- 2D coordinate frame transformations
- Hierarchical frame support
- Point and Vector classes with proper transformation semantics
- Visualization module (optional, requires matplotlib)
- Comprehensive examples and documentation