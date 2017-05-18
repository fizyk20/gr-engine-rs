use diffgeom::coordinates::{CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::Vector;
use generic_array::{ArrayLength, GenericArray};
use numeric::StateVector;
use numeric_algs::State;
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

impl<C: CoordinateSystem> Clone for Particle<C>
    where C::Dimension: Pow<U1>,
          Exp<C::Dimension, U1>: ArrayLength<f64>
{
    fn clone(&self) -> Self {
        Particle {
            x: self.x.clone(),
            v: self.v.clone(),
        }
    }
}

impl<C: CoordinateSystem> State for Particle<C>
    where C::Dimension: Pow<U1> + Mul<U2> + Unsigned,
          Exp<C::Dimension, U1>: ArrayLength<f64>,
          Prod<C::Dimension, U2>: ArrayLength<f64>,
          <Prod<C::Dimension, U2> as ArrayLength<f64>>::ArrayType: Copy
{
    type Derivative = StateVector<Prod<C::Dimension, U2>>;

    fn shift(&self, dir: Self::Derivative, amount: f64) -> Self {
        let d = C::Dimension::to_usize();
        let mut new_x = GenericArray::default();
        let mut new_v = GenericArray::default();
        for i in 0..d {
            new_x[i] = self.x[i] + dir.0[i] * amount;
            new_v[i] = self.v[i] + dir.0[i + d] * amount;
        }
        let x = Point::new(new_x);
        let v = Vector::new(x.clone(), new_v);
        Particle { x: x, v: v }
    }
}

impl<C: CoordinateSystem> DiffEq<Particle<C>> for Particle<C>
    where C::Dimension: Pow<U1> + Mul<U2> + Unsigned + Pow<U2> + Pow<U3>,
          Exp<C::Dimension, U1>: ArrayLength<f64>,
          Prod<C::Dimension, U2>: ArrayLength<f64>,
          <Prod<C::Dimension, U2> as ArrayLength<f64>>::ArrayType: Copy,
          C: MetricSystem,
          Exp<C::Dimension, U2>: ArrayLength<f64>,
          Exp<C::Dimension, U3>: ArrayLength<f64>
{
    fn derivative(&self, state: Particle<C>) -> StateVector<Prod<C::Dimension, U2>> {
        let christoffel = C::christoffel(&state.x);
        let temp = inner!(_, Vector<C>; U1, U3; christoffel, state.v.clone());
        let cov_der = inner!(_, Vector<C>; U1, U2; temp, state.v.clone());
        let mut result = GenericArray::default();
        let d = C::Dimension::to_usize();
        for i in 0..d {
            result[i] = state.v[i];
            result[i + d] = -cov_der[i];
        }
        StateVector(result)
    }
}
