use super::state_vector::StateVector;
use generic_array::ArrayLength;

pub trait DiffEq<N>: Fn(&StateVector<N>) -> StateVector<N>
    where N: ArrayLength<f64>,
          N::ArrayType: Copy
{
    fn derivative(&self, StateVector<N>) -> StateVector<N>;
}

pub enum StepSize {
    UseDefault,
    Step(f64),
}

pub trait Integrator<N>
    where N: ArrayLength<f64>,
          N::ArrayType: Copy
{
    fn propagate<D>(&mut self,
                    start: StateVector<N>,
                    diff_eq: D,
                    step: StepSize)
                    -> StateVector<N>
        where D: DiffEq<N>;
}
