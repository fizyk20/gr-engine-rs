use diffgeom::coordinates::{CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::Vector;
use generic_array::{ArrayLength, GenericArray};
use numeric::StateVector;
use numeric_algs::State;
use std::ops::Mul;
use typenum::consts::{U1, U2, U3};
use typenum::{Exp, Pow, Prod, Unsigned};

pub struct Particle<C: CoordinateSystem>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    x: Point<C>,
    v: Vector<C>,
}

impl<C: CoordinateSystem> Clone for Particle<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    fn clone(&self) -> Self {
        Particle {
            x: self.x.clone(),
            v: self.v.clone(),
        }
    }
}

impl<C: CoordinateSystem> Particle<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    pub fn new(x: Point<C>, v: Vector<C>) -> Self {
        Particle { x: x, v: v }
    }

    pub fn get_pos(&self) -> &Point<C> {
        &self.x
    }

    pub fn get_vel(&self) -> &Vector<C> {
        &self.v
    }
}

impl<C: CoordinateSystem> State for Particle<C>
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
            self.v[i] += dir.0[i + d] * amount;
            self.v.set_point(self.x.clone());
        }
    }
}

impl<C: CoordinateSystem> Particle<C>
where
    C::Dimension: Pow<U1> + Mul<U2> + Unsigned + Pow<U2> + Pow<U3>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
    Prod<C::Dimension, U2>: ArrayLength<f64>,
    <Prod<C::Dimension, U2> as ArrayLength<f64>>::ArrayType: Copy,
    C: MetricSystem,
    Exp<C::Dimension, U2>: ArrayLength<f64>,
    Exp<C::Dimension, U3>: ArrayLength<f64>,
{
    pub fn derivative(&self) -> StateVector<Prod<C::Dimension, U2>> {
        let christoffel = C::christoffel(&self.x);
        let temp = inner!(_, Vector<C>; U1, U3; christoffel, self.v.clone());
        let cov_der = inner!(_, Vector<C>; U1, U2; temp, self.v.clone());
        let mut result = GenericArray::default();
        let d = C::Dimension::to_usize();
        for i in 0..d {
            result[i] = self.v[i];
            result[i + d] = -cov_der[i];
        }
        StateVector(result)
    }
}
