use crate::numeric::StateVector;
use crate::typenum::consts::{U1, U2, U3};
use crate::typenum::{Exp, Pow, Prod, Same, Unsigned};
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::inner;
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::Vector;
use generic_array::{ArrayLength, GenericArray};
use numeric_algs::State;
use std::ops::Mul;

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

impl<C: CoordinateSystem> Particle<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    pub fn convert<C2: CoordinateSystem + 'static>(&self) -> Particle<C2>
    where
        C: ConversionTo<C2>,
        C2::Dimension: Pow<U1> + Pow<U2>,
        Exp<C2::Dimension, U1>: ArrayLength<f64>,
        Exp<C2::Dimension, U2>: ArrayLength<f64>,
        C2::Dimension: Same<C::Dimension>,
    {
        let new_x: Point<C2> = C::convert_point(&self.x);
        let new_v: Vector<C2> = self.v.convert();
        Particle { x: new_x, v: new_v }
    }
}

pub trait PosAndVel<D: Unsigned + ArrayLength<f64> + Pow<U1>>
where
    Exp<D, U1>: ArrayLength<f64>,
{
    fn get_pos(&self) -> &GenericArray<f64, D>;
    fn get_vel(&self) -> &GenericArray<f64, Exp<D, U1>>;
}

impl<C: CoordinateSystem> PosAndVel<C::Dimension> for Particle<C>
where
    C::Dimension: Pow<U1>,
    Exp<C::Dimension, U1>: ArrayLength<f64>,
{
    fn get_pos(&self) -> &GenericArray<f64, C::Dimension> {
        self.x.coords_array()
    }

    fn get_vel(&self) -> &GenericArray<f64, Exp<C::Dimension, U1>> {
        self.v.coords_array()
    }
}
