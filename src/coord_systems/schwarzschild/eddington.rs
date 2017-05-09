use typenum::consts::U4;
use diffgeom::coordinates::{CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{TwoForm, InvTwoForm};
use super::Mass;
use std::marker::PhantomData;

pub struct EddingtonFinkelstein<M: Mass> {
    _m: PhantomData<M>,
}

impl<M: Mass> CoordinateSystem for EddingtonFinkelstein<M> {
    type Dimension = U4;
}

impl<M: Mass> MetricSystem for EddingtonFinkelstein<M> {
    fn g(x: &Point<Self>) -> TwoForm<Self> {
        let u = x[0];
        let r = x[1];
        let th = x[2];
        let ph = x[3];
        let m = M::mass();
        TwoForm::new(x.clone(),
                     arr![f64; 
            1.0 - 2.0*m/r, -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -r*r, 0.0,
            0.0, 0.0, 0.0, -r*r*th.sin()*th.sin()
        ])
    }

    fn inv_g(x: &Point<Self>) -> InvTwoForm<Self> {
        let u = x[0];
        let r = x[1];
        let th = x[2];
        let ph = x[3];
        let m = M::mass();
        InvTwoForm::new(x.clone(),
                        arr![f64; 
            -1.0 + 2.0*m/r, -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0/(r*r), 0.0,
            0.0, 0.0, 0.0, -1.0/(r*r*th.sin()*th.sin())
        ])
    }

    // TODO
    //fn dg(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
    //fn covariant_christoffel(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
    //fn christoffel(x: &Point<Self>) -> Tensor<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))>
}
