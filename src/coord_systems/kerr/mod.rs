mod eddington;
mod polar_eddington;

pub trait Properties {
    fn mass() -> f64;
    fn ang_momentum() -> f64;
}

pub use self::eddington::EddingtonFinkelstein;
pub use self::polar_eddington::{NearPole0EF, NearPolePiEF};
