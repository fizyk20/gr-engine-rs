mod schwarzschild;
mod polar_schwarzschild;
mod eddington;
mod polar_eddington;

pub trait Mass {
    fn mass() -> f64;
}

pub use self::eddington::EddingtonFinkelstein;
