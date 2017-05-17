use generic_array::{GenericArray, ArrayLength};
use std::ops::{Add, Sub, Mul, Div};

pub struct StateVector<N: ArrayLength<f64>>(GenericArray<f64, N>);

impl<N: ArrayLength<f64>> StateVector<N> {
    pub fn abs(&self) -> f64 {
        0.0
    }
}

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
