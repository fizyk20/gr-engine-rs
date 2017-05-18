use generic_array::{ArrayLength, GenericArray};
use numeric_algs::{State, StateDerivative};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub struct StateVector<N: ArrayLength<f64>>(pub GenericArray<f64, N>);

impl<N: ArrayLength<f64>> Clone for StateVector<N>
    where N::ArrayType: Clone
{
    fn clone(&self) -> Self {
        StateVector(self.0.clone())
    }
}

impl<N: ArrayLength<f64>> Copy for StateVector<N> where N::ArrayType: Copy {}

impl<N: ArrayLength<f64>> Add<StateVector<N>> for StateVector<N>
    where N::ArrayType: Copy
{
    type Output = StateVector<N>;

    fn add(mut self, other: StateVector<N>) -> StateVector<N> {
        for i in 0..N::to_usize() {
            self.0[i] += other.0[i];
        }
        self
    }
}

impl<N: ArrayLength<f64>> Sub<StateVector<N>> for StateVector<N>
    where N::ArrayType: Copy
{
    type Output = StateVector<N>;

    fn sub(mut self, other: StateVector<N>) -> StateVector<N> {
        for i in 0..N::to_usize() {
            self.0[i] -= other.0[i];
        }
        self
    }
}

impl<N: ArrayLength<f64>> Mul<f64> for StateVector<N>
    where N::ArrayType: Copy
{
    type Output = StateVector<N>;

    fn mul(mut self, other: f64) -> StateVector<N> {
        for i in 0..N::to_usize() {
            self.0[i] *= other;
        }
        self
    }
}

impl<N: ArrayLength<f64>> Mul<StateVector<N>> for f64
    where N::ArrayType: Copy
{
    type Output = StateVector<N>;

    fn mul(self, mut other: StateVector<N>) -> StateVector<N> {
        for i in 0..N::to_usize() {
            other.0[i] *= self;
        }
        other
    }
}

impl<N: ArrayLength<f64>> Div<f64> for StateVector<N>
    where N::ArrayType: Copy
{
    type Output = StateVector<N>;

    fn div(mut self, other: f64) -> StateVector<N> {
        for i in 0..N::to_usize() {
            self.0[i] /= other;
        }
        self
    }
}

impl<N: ArrayLength<f64>> Neg for StateVector<N>
    where N::ArrayType: Copy
{
    type Output = StateVector<N>;

    fn neg(mut self) -> StateVector<N> {
        for i in 0..N::to_usize() {
            self.0[i] = -self.0[i];
        }
        self
    }
}

impl<N: ArrayLength<f64>> StateDerivative for StateVector<N>
    where N::ArrayType: Copy
{
    fn abs(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..N::to_usize() {
            sum += self.0[i] * self.0[i];
        }
        sum.sqrt()
    }
}

impl<N: ArrayLength<f64>> State for StateVector<N>
    where N::ArrayType: Copy
{
    type Derivative = StateVector<N>;
    fn shift(&self, dir: Self::Derivative, amount: f64) -> Self {
        *self + dir * amount
    }
}
