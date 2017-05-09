use super::state_vector::StateVector;
use generic_array::ArrayLength;

pub trait Integrator<N>: Fn(&StateVector<N>) -> StateVector<N>
    where N: ArrayLength<f64>,
          N::ArrayType: Copy
{
    fn derivative(&StateVector<N>) -> StateVector<N>;
}
