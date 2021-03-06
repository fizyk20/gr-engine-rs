use crate::numeric::StateVector;
use crate::typenum::consts::{U0, U1, U2, U3, U5};
use crate::typenum::{Exp, Pow, Prod, Same, Unsigned};
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::inner;
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::Vector;
use generic_array::{ArrayLength, GenericArray};
use numeric_algs::State;
use std::ops::Mul;

pub struct Entity<C: CoordinateSystem>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    // position
    x: Point<C>,
    // the local basis of spacetime; the vectors represent:
    // 0: 4-velocity
    // 1: the forward direction
    // 2: the right direction
    // 3: the up direction
    dirs: [Vector<C>; 4],
    // applied force in local coordinates
    force: [f64; 3],
    // angular velocity in local coordinates
    ang_vel: [f64; 3],
}

impl<C: CoordinateSystem> Clone for Entity<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    fn clone(&self) -> Self {
        Entity {
            x: self.x.clone(),
            dirs: self.dirs.clone(),
            force: self.force,
            ang_vel: self.ang_vel,
        }
    }
}

impl<C: CoordinateSystem> Entity<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    pub fn new(
        x: Point<C>,
        v: Vector<C>,
        local_x: Vector<C>,
        local_y: Vector<C>,
        local_z: Vector<C>,
    ) -> Self {
        Entity {
            x: x,
            dirs: [v, local_x, local_y, local_z],
            force: [0.0; 3],
            ang_vel: [0.0; 3],
        }
    }

    pub fn orthonormalize(&mut self)
    where
        C: MetricSystem,
        C::Dimension: Pow<U2> + Pow<U3>,
        Exp<C::Dimension, U2>: ArrayLength<f64>,
        Exp<C::Dimension, U3>: ArrayLength<f64>,
    {
        let g = C::g(&self.x);
        for i in 0..4 {
            let dirs_i_cov = inner!(_, Vector<C>; U1, U2; g.clone(), self.dirs[i].clone());
            for j in 0..i {
                let dot_ij =
                    *inner!(_, Vector<C>; U0, U1; dirs_i_cov.clone(), self.dirs[j].clone());
                let abs_j_temp = inner!(_, Vector<C>; U1, U2; g.clone(), self.dirs[j].clone());
                let abs_j = *inner!(_, Vector<C>; U0, U1; abs_j_temp, self.dirs[j].clone());
                self.dirs[i] -= self.dirs[j].clone() * dot_ij / abs_j;
            }
            let dirs_i_abs = *inner!(_, Vector<C>; U0, U1; dirs_i_cov, self.dirs[i].clone());
            self.dirs[i] /= dirs_i_abs.abs().sqrt();
        }
    }

    pub fn add_force(&mut self, x: f64, y: f64, z: f64) {
        self.force[0] += x;
        self.force[1] += y;
        self.force[2] += z;
    }

    pub fn add_ang_vel(&mut self, x: f64, y: f64, z: f64) {
        self.ang_vel[0] += x;
        self.ang_vel[1] += y;
        self.ang_vel[2] += z;
    }

    pub fn reset_force(&mut self) {
        self.force = [0.0; 3];
    }

    pub fn reset_ang_vel(&mut self) {
        self.ang_vel = [0.0; 3];
    }

    pub fn get_pos(&self) -> &Point<C> {
        &self.x
    }

    pub fn get_vel(&self) -> &Vector<C> {
        &self.dirs[0]
    }

    // calculates the covariant derivative of the selected component of the local basis in the
    // direction of 4-velocity
    fn calculate_derivative(&self, dir: usize) -> Vector<C> {
        let gen_matrix = [
            [0.0, self.force[0], self.force[1], self.force[2]],
            [self.force[0], 0.0, -self.ang_vel[2], self.ang_vel[1]],
            [self.force[1], self.ang_vel[2], 0.0, -self.ang_vel[0]],
            [self.force[2], -self.ang_vel[1], self.ang_vel[0], 0.0],
        ];

        let mut result = Vector::zero(self.x.clone());

        for i in 0..4 {
            result += gen_matrix[i][dir] * self.dirs[i].clone();
        }

        result
    }
}

impl<C: CoordinateSystem> State for Entity<C>
where
    C::Dimension: Pow<U1> + Mul<U2> + Unsigned,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
    Prod<C::Dimension, U2>: ArrayLength<f64>,
    <Prod<C::Dimension, U2> as ArrayLength<f64>>::ArrayType: Copy,
{
    type Derivative = StateVector<Prod<C::Dimension, U2>>;

    fn shift_in_place(&mut self, dir: &Self::Derivative, amount: f64) {
        let d = C::Dimension::to_usize();
        for i in 0..d {
            self.x[i] += dir.0[i] * amount;
            for j in 0..4 {
                self.dirs[j][i] += dir.0[i + (j + 1) * d] * amount;
                self.dirs[j].set_point(self.x.clone());
            }
        }
    }
}

impl<C: CoordinateSystem> Entity<C>
where
    C::Dimension: Pow<U1> + Mul<U5> + Unsigned + Pow<U2> + Pow<U3>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
    Prod<C::Dimension, U5>: ArrayLength<f64>,
    <Prod<C::Dimension, U5> as ArrayLength<f64>>::ArrayType: Copy,
    C: MetricSystem,
    Exp<C::Dimension, U2>: ArrayLength<f64>,
    Exp<C::Dimension, U3>: ArrayLength<f64>,
{
    pub fn derivative(&self) -> StateVector<Prod<C::Dimension, U5>> {
        let christoffel = C::christoffel(&self.x);
        let chr_times_v = inner!(_, Vector<C>; U1, U3; christoffel, self.dirs[0].clone());
        let mut result = GenericArray::default();
        let d = C::Dimension::to_usize();

        // derivative of directions
        for j in 0..4 {
            let chr_part = inner!(_, Vector<C>; U1, U2; chr_times_v.clone(), self.dirs[j].clone());
            let cov_der = self.calculate_derivative(j);

            for i in 0..d {
                result[i + (j + 1) * d] = cov_der[i] - chr_part[i];
            }
        }

        // position derivative
        for i in 0..d {
            result[i] = self.dirs[0][i];
        }
        StateVector(result)
    }
}

impl<C: CoordinateSystem> Entity<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    pub fn convert<C2: CoordinateSystem + 'static>(&self) -> Entity<C2>
    where
        C: ConversionTo<C2>,
        C2::Dimension: Pow<U1> + Pow<U2>,
        Exp<C2::Dimension, U1>: ArrayLength<f64>,
        Exp<C2::Dimension, U2>: ArrayLength<f64>,
        C2::Dimension: Same<C::Dimension>,
    {
        let new_x: Point<C2> = C::convert_point(&self.x);
        let new_dirs: [Vector<C2>; 4] = [
            self.dirs[0].convert(),
            self.dirs[1].convert(),
            self.dirs[2].convert(),
            self.dirs[3].convert(),
        ];
        Entity {
            x: new_x,
            dirs: new_dirs,
            force: self.force,
            ang_vel: self.ang_vel,
        }
    }
}

pub trait PosAndVel<D: Unsigned + ArrayLength<f64> + Pow<U1>>
where
    Exp<D, U1>: ArrayLength<f64>,
{
    fn get_pos(&self) -> &GenericArray<f64, D>;
    fn get_vel(&self) -> &GenericArray<f64, Exp<D, U1>>;
}

impl<C: CoordinateSystem> PosAndVel<C::Dimension> for Entity<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    fn get_pos(&self) -> &GenericArray<f64, C::Dimension> {
        self.x.coords_array()
    }

    fn get_vel(&self) -> &GenericArray<f64, Exp<C::Dimension, U1>> {
        self.dirs[0].coords_array()
    }
}
