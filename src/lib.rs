// Needed for Christoffel symbols in arr! macro
// (64 numbers in a 4D spacetime)
#![recursion_limit = "70"]

#[macro_use]
pub extern crate diffgeom;
extern crate numeric_algs;
#[macro_use]
pub extern crate generic_array;

pub use generic_array::typenum;

pub mod coord_systems;
pub mod numeric;
pub mod particle;
