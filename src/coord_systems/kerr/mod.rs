mod eddington;

pub trait Properties {
    fn mass() -> f64;
    fn ang_momentum() -> f64;
}

pub use self::eddington::EddingtonFinkelstein;
