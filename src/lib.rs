// Needed for Christoffel symbols in arr! macro
// (64 numbers in a 4D spacetime)
#![recursion_limit = "70"]

pub extern crate diffgeom;
pub extern crate generic_array;
extern crate numeric_algs;

pub use generic_array::typenum;

pub mod coord_systems;
mod entity;
pub mod numeric;
mod particle;

pub use crate::entity::Entity;
pub use crate::particle::{Particle, PosAndVel};
