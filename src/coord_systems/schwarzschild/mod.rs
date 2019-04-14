mod eddington;
mod polar_eddington;
mod polar_schwarzschild;
mod schwarzschild;

pub trait Mass {
    fn mass() -> f64;
}

pub use self::eddington::EddingtonFinkelstein;
pub use self::polar_eddington::{NearPole0EF, NearPolePiEF};
pub use self::polar_schwarzschild::{NearPole0Schw, NearPolePiSchw};
pub use self::schwarzschild::Schwarzschild;
