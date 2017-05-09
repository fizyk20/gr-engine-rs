use typenum::consts::U4;
use diffgeom::coordinates::{CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{CovariantIndex, ContravariantIndex, TwoForm, InvTwoForm, Tensor};
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

    fn christoffel(x: &Point<Self>)
                   -> Tensor<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))> {
        let u = x[0];
        let r = x[1];
        let th = x[2];
        let ph = x[3];
        let m = M::mass();
        Tensor::<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))>::new(x.clone(),
                                                                                    arr![f64;
            m/r/r, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -r, 0.0,
            0.0, 0.0, 0.0, -r*th.sin()*th.sin(),

            m*(1.0-2.0*m/r)/r/r, -m/r/r, 0.0, 0.0,
            -m/r/r, 0.0, 0.0, 0.0,
            0.0, 0.0, -(r-2.0*m), 0.0,
            0.0, 0.0, 0.0, -(r-2.0*m)*th.sin()*th.sin(),

            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0/r, 0.0,
            0.0, 1.0/r, 0.0, 0.0,
            0.0, 0.0, 0.0, -th.sin()*th.cos(),

            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0/r,
            0.0, 0.0, 0.0, th.cos()/th.sin(),
            0.0, 1.0/r, th.cos()/th.sin(), 0.0
        ])
    }

    // TODO
    //fn dg(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
    //fn covariant_christoffel(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
}
