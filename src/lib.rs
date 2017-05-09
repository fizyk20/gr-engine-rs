// Needed for Christoffel symbols in arr! macro
// (64 numbers in a 4D spacetime)
#![recursion_limit="70"]

#[macro_use]
extern crate generic_array;
extern crate diffgeom;
extern crate typenum;

pub mod numeric;
pub mod coord_systems;
