mod eddington;
mod polar_eddington;
mod polar_schwarzschild;
mod schwarzschild;

pub trait Mass {
    fn mass() -> f64;
}

pub use self::eddington::EddingtonFinkelstein;
