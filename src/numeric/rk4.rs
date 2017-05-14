use super::{StepSize, Integrator, DiffEq, StateVector};
use generic_array::ArrayLength;

pub struct RK4Integrator {
    default_step: f64,
}

impl RK4Integrator {
    pub fn new(step_size: f64) -> Self {
        RK4Integrator { default_step: step_size }
    }
}

impl Integrator for RK4Integrator {
    fn propagate<N, D>(&mut self,
                       start: StateVector<N>,
                       diff_eq: D,
                       step_size: StepSize)
                       -> StateVector<N>
        where N: ArrayLength<f64>,
              N::ArrayType: Copy,
              D: DiffEq<N>
    {
        let h = match step_size {
            StepSize::UseDefault => self.default_step,
            StepSize::Step(x) => x,
        };

        let k1 = diff_eq.derivative(start);
        let k2 = diff_eq.derivative(start + k1 * (h / 2.0));
        let k3 = diff_eq.derivative(start + k2 * (h / 2.0));
        let k4 = diff_eq.derivative(start + k3 * h);

        start + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * h / 6.0
    }
}
