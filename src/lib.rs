// Needed for Christoffel symbols in arr! macro
// (64 numbers in a 4D spacetime)
#![recursion_limit = "70"]

pub extern crate diffgeom;
pub extern crate generic_array;
extern crate numeric_algs;

pub use generic_array::typenum;

pub mod coord_systems;
pub mod numeric;
pub mod particle;
