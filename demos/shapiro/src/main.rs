extern crate diffgeom;
extern crate gr_engine;
#[macro_use]
extern crate generic_array;
extern crate numeric_algs;

use diffgeom::coordinates::Point;
use diffgeom::tensors::Vector;
use gr_engine::coord_systems::schwarzschild::{Mass, Schwarzschild};
use gr_engine::particle::Particle;
use numeric_algs::integration::{DPIntegrator, Integrator, StepSize};
use std::f64::consts::PI;

struct Sun;
impl Mass for Sun {
    fn mass() -> f64 {
        M
    }
}

type Coords = Schwarzschild<Sun>;

const M: f64 = 4.9e-6; // mass of the Sun in seconds, with c = G = 1
const D: f64 = 2.33; // radius of the Sun in seconds
const YE: f64 = 498.67; // Earth "y" coordinate in seconds
const YV: f64 = 370.7; // Venus "y" coordinate in seconds

fn propagate_single_photon(
    photon: &mut Particle<Coords>,
    integrator: &mut DPIntegrator<Particle<Coords>>,
    r_max: f64,
) -> f64 {
    let mut last_pos = photon.get_pos().clone();

    let mut i = 1;
    while photon.get_pos()[1] < r_max {
        last_pos = photon.get_pos().clone();
        integrator.propagate_in_place(photon, Particle::derivative, StepSize::UseDefault);
        i += 1;
        if i % 100 == 0 {
            println!("Iteration {}... r = {}", i, photon.get_pos()[1]);
        }
    }

    let pos = photon.get_pos();
    let (t, r) = (pos[0], pos[1]);
    let (last_t, last_r) = (last_pos[0], last_pos[1]);

    let coeff = (r_max - last_r) / (r - last_r);

    last_t + (t - last_t) * coeff
}

fn main() {
    let u0 = (D * D * D / (D - M * 2.0)).sqrt();
    let r_e = (D * D + YE * YE).sqrt();
    let r_v = (D * D + YV * YV).sqrt();

    let t_flat = 2.0 * (YE + YV);

    let start_point = Point::<Coords>::new(arr![f64; 0.0, D, PI / 2.0, 0.0]);
    let u_init1 = Vector::<Coords>::new(start_point.clone(), arr![f64; u0, 0.0, 0.0, 1.0]);
    let u_init2 = Vector::<Coords>::new(start_point.clone(), arr![f64; -u0, 0.0, 0.0, -1.0]);

    let mut photon1 = Particle::new(start_point.clone(), u_init1);
    let mut photon2 = Particle::new(start_point.clone(), u_init2);

    let mut integrator = DPIntegrator::<Particle<Coords>>::new(0.01, 0.0001, 0.1, 1e-12);

    println!("Propagating the first photon...");
    let t1 = propagate_single_photon(&mut photon1, &mut integrator, r_e);

    integrator.reset();

    println!("Propagating the second photon...");
    let t2 = propagate_single_photon(&mut photon2, &mut integrator, r_v);

    println!("Propagation finished.");
    println!("t1 = {}", t1);
    println!("t2 = {}", t2);

    let dt = (t1 - t2) * 2.0 * (1.0 - 2.0 * M / r_e).sqrt();
    println!("dt = {}", dt);
    println!("delay = {}", dt - t_flat);
}
