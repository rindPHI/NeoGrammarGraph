# Changelog

## [unreleased]

## [0.2.2] - 2023-09-27

### Added

- The canonical version of a grammar can be obtained via the :code:`canonical_grammar`
  field. The canonical grammar is computed lazily; no performance overhead is to be
  expected from this change.
 
## [0.2.1] - 2023-05-11

### Changed

- Added caching to reachability and shortest paths.
  
## [0.2.0] - 2023-05-11

### Changed

- Bug fix in computation of k-paths; before, the start symbol was not correctly handled
  and k-paths for reference nodes were faulty.
- One can now specify multiple start nodes for the computation of k-paths. The parameter
  `start_node` has become `start_nodes`.

## [0.1.1] - 2023-05-03

### Added

- Validity check for derivation trees.

### Changed

- Corrected representation of grammars with epsilon expansions.

## [0.1.0] - 2023-05-03

### Added

- Converting derivation trees to graph-tool graphs
- k-path computation for derivation trees

### Changed

- Added a proper type hierarchy for graph nodes (not only structured strings).

## [0.0.5] - 2023-04-21

### Added

- k-path computation.
- Subgraph extraction.
- Filtering nodes of a graph.

## [0.0.4] - 2023-04-19

### Added

- Computation of all paths between two grammar symbols.

## [0.0.2] - 2023-04-19

### Added

- Shortest path computation.

## [0.0.1] - 2023-04-18

### Added
## [0.0.3] - 2023-04-19

### Added

- Access to the children of a grammar symbol in the graph.

## [0.0.2] - 2023-04-19

### Added

- Shortest path computation.

## [0.0.1] - 2023-04-18

### Added

- Initial commit; functionality for checking reachability and exporting to DOT.
