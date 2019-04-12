extern crate diffgeom;
extern crate gr_engine;
#[macro_use]
extern crate generic_array;
extern crate numeric_algs;

use diffgeom::coordinates::Point;
use diffgeom::tensors::Vector;
use gr_engine::coord_systems::schwarzschild::{EddingtonFinkelstein, Mass};
use gr_engine::numeric::StateVector;
use gr_engine::particle::Particle;
use numeric_algs::integration::{DPIntegrator, Integrator, StepSize};
use std::f64::consts::PI;

struct Sun;
impl Mass for Sun {
    fn mass() -> f64 {
        M
    }
}

type Coords = EddingtonFinkelstein<Sun>;

const M: f64 = 4.9e-6; // mass of the Sun in seconds, with c = G = 1
const D: f64 = 2.33; // radius of the Sun in seconds
const YE: f64 = 498.67; // Earth "y" coordinate in seconds
const YV: f64 = 370.7; // Venus "y" coordinate in seconds

fn u(t: f64, r: f64) -> f64 {
    t + r + (0.5 * (r - 2.0 * M) / M).ln() * 2.0 * M
}

fn t(u: f64, r: f64) -> f64 {
    u - r - (0.5 * (r - 2.0 * M) / M).ln() * 2.0 * M
}

fn main() {
    let u0 = (D * D * D / (D - M * 2.0)).sqrt();
    let r_e = (D * D + YE * YE).sqrt();
    let r_v = (D * D + YV * YV).sqrt();

    let t_flat = 2.0 * (YE + YV);

    let start_point = Point::<Coords>::new(arr![f64; u(0.0, D), D, PI / 2.0, 0.0]);
    let u_init1 = Vector::<Coords>::new(start_point.clone(), arr![f64; u0, 0.0, 0.0, 1.0]);
    let u_init2 = Vector::<Coords>::new(start_point.clone(), arr![f64; -u0, 0.0, 0.0, -1.0]);

    let mut photon1 = Particle::new(start_point.clone(), u_init1);
    let mut photon2 = Particle::new(start_point.clone(), u_init2);

    let mut integrator1 = DPIntegrator::<Particle<Coords>>::new(0.01, 0.0001, 0.1, 1e-12);
    let mut integrator2 = DPIntegrator::<Particle<Coords>>::new(0.01, 0.0001, 0.1, 1e-12);

    let mut last_pos = photon1.get_pos().clone();
    println!("Propagating the first photon...");

    let mut i = 1;
    while photon1.get_pos()[1] < r_e {
        last_pos = photon1.get_pos().clone();
        integrator1.propagate_in_place(&mut photon1, Particle::derivative, StepSize::UseDefault);
        i += 1;
        if i % 100 == 0 {
            println!("Iteration {}... r = {}", i, photon1.get_pos()[1]);
        }
    }
    let pos = photon1.get_pos().clone();
    let pos = StateVector(arr![f64; pos[0], pos[1], pos[2], pos[3]]);
    let last_pos = StateVector(arr![f64; last_pos[0], last_pos[1], last_pos[2], last_pos[3]]);
    let coeff = (r_e - last_pos.0[1]) / (pos.0[1] - last_pos.0[1]);
    let last_pos = last_pos + (pos - last_pos) * coeff;
    let t1 = t(last_pos.0[0], last_pos.0[1]);

    let mut last_pos = photon2.get_pos().clone();
    println!("Propagating the second photon...");

    let mut i = 1;
    while photon2.get_pos()[1] < r_v {
        last_pos = photon2.get_pos().clone();
        integrator2.propagate_in_place(&mut photon2, Particle::derivative, StepSize::UseDefault);
        i += 1;
        if i % 100 == 0 {
            println!("Iteration {}... r = {}", i, photon2.get_pos()[1]);
        }
    }
    let pos = photon2.get_pos().clone();
    let pos = StateVector(arr![f64; pos[0], pos[1], pos[2], pos[3]]);
    let last_pos = StateVector(arr![f64; last_pos[0], last_pos[1], last_pos[2], last_pos[3]]);
    let coeff = (r_v - last_pos.0[1]) / (pos.0[1] - last_pos.0[1]);
    let last_pos = last_pos + (pos - last_pos) * coeff;
    let t2 = t(last_pos.0[0], last_pos.0[1]);

    println!("Propagation finished.");
    println!("t1 = {}", t1);
    println!("t2 = {}", t2);

    let dt = (t1 - t2) * 2.0 * (1.0 - 2.0 * M / r_e).sqrt();
    println!("dt = {}", dt);
    println!("delay = {}", dt - t_flat);
}
