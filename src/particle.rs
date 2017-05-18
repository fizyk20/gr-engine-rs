use diffgeom::coordinates::{CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::Vector;
use generic_array::{ArrayLength, GenericArray};
use numeric::StateVector;
use numeric_algs::integration::DiffEq;
use std::ops::Mul;
use typenum::{Exp, Pow, Prod, Unsigned};
use typenum::consts::{U1, U2, U3};

struct Particle<C: CoordinateSystem>
    where C::Dimension: Pow<U1>,
          Exp<C::Dimension, U1>: ArrayLength<f64>
{
    x: Point<C>,
    v: Vector<C>,
}

impl<C: CoordinateSystem> From<Particle<C>> for StateVector<Prod<C::Dimension, U2>>
    where C::Dimension: Pow<U1> + Mul<U2> + Unsigned,
          Exp<C::Dimension, U1>: ArrayLength<f64>,
          Prod<C::Dimension, U2>: ArrayLength<f64>
{
    fn from(particle: Particle<C>) -> Self {
        let mut result = GenericArray::default();
        let d = C::Dimension::to_usize();
        for i in 0..d {
            result[i] = particle.x[i];
        }
        for i in 0..d {
            result[i + d] = particle.v[i];
        }
        StateVector(result)
    }
}

impl<C: CoordinateSystem> From<StateVector<Prod<C::Dimension, U2>>> for Particle<C>
    where C::Dimension: Pow<U1> + Mul<U2> + Unsigned,
          Exp<C::Dimension, U1>: ArrayLength<f64>,
          Prod<C::Dimension, U2>: ArrayLength<f64>
{
    fn from(state_vec: StateVector<Prod<C::Dimension, U2>>) -> Self {
        let d = C::Dimension::to_usize();
        let x = Point::from_slice(&state_vec.0[0..d]);
        let v = Vector::from_slice(x.clone(), &state_vec.0[d..d * 2]);
        Particle { x: x, v: v }
    }
}

impl<C: CoordinateSystem> DiffEq<StateVector<Prod<C::Dimension, U2>>> for Particle<C>
    where C::Dimension: Pow<U1> + Mul<U2> + Unsigned + Pow<U2> + Pow<U3>,
          Exp<C::Dimension, U1>: ArrayLength<f64>,
          Prod<C::Dimension, U2>: ArrayLength<f64>,
          <Prod<C::Dimension, U2> as ArrayLength<f64>>::ArrayType: Copy,
          C: MetricSystem,
          Exp<C::Dimension, U2>: ArrayLength<f64>,
          Exp<C::Dimension, U3>: ArrayLength<f64>
{
    fn derivative(&self,
                  state: StateVector<Prod<C::Dimension, U2>>)
                  -> StateVector<Prod<C::Dimension, U2>> {
        let particle = Particle::<C>::from(state);
        let christoffel = C::christoffel(&particle.x);
        let temp = inner!(_, Vector<C>; U1, U3; christoffel, particle.v.clone());
        let cov_der = inner!(_, Vector<C>; U1, U2; temp, particle.v.clone());
        let mut result = GenericArray::default();
        let d = C::Dimension::to_usize();
        for i in 0..d {
            result[i] = particle.v[i];
        }
        for i in 0..d {
            result[i + d] = -cov_der[i];
        }
        StateVector(result)
    }
}
