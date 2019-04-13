use super::{Mass, Schwarzschild};
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{ContravariantIndex, CovariantIndex, InvTwoForm, Matrix, Tensor, TwoForm};
use std::marker::PhantomData;
use typenum::consts::U4;

pub struct EddingtonFinkelstein<M: Mass> {
    _m: PhantomData<M>,
}

impl<M: Mass> CoordinateSystem for EddingtonFinkelstein<M> {
    type Dimension = U4;
}

impl<M: Mass> MetricSystem for EddingtonFinkelstein<M> {
    fn g(x: &Point<Self>) -> TwoForm<Self> {
        let r = x[1];
        let th = x[2];
        let m = M::mass();
        TwoForm::new(
            x.clone(),
            arr![f64;
                1.0 - 2.0*m/r, -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -r*r, 0.0,
                0.0, 0.0, 0.0, -r*r*th.sin()*th.sin()
            ],
        )
    }

    fn inv_g(x: &Point<Self>) -> InvTwoForm<Self> {
        let r = x[1];
        let th = x[2];
        let m = M::mass();
        InvTwoForm::new(
            x.clone(),
            arr![f64;
                -1.0 + 2.0*m/r, -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.0/(r*r), 0.0,
                0.0, 0.0, 0.0, -1.0/(r*r*th.sin()*th.sin())
            ],
        )
    }

    fn christoffel(
        x: &Point<Self>,
    ) -> Tensor<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))> {
        let r = x[1];
        let th = x[2];
        let m = M::mass();
        let mr = m / r;
        let r2m = r - 2.0 * m;
        Tensor::<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))>::new(
            x.clone(),
            arr![f64;
                mr/r, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -r, 0.0,
                0.0, 0.0, 0.0, -r*th.sin()*th.sin(),

                m*r2m/r/r/r, -mr/r, 0.0, 0.0,
                -mr/r, 0.0, 0.0, 0.0,
                0.0, 0.0, -r2m, 0.0,
                0.0, 0.0, 0.0, -r2m*th.sin()*th.sin(),

                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0/r, 0.0,
                0.0, 1.0/r, 0.0, 0.0,
                0.0, 0.0, 0.0, -th.sin()*th.cos(),

                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0/r,
                0.0, 0.0, 0.0, th.cos()/th.sin(),
                0.0, 1.0/r, th.cos()/th.sin(), 0.0
            ],
        )
    }

    // TODO
    //fn dg(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
    //fn covariant_christoffel(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
}

// Conversions

impl<M: Mass + 'static> ConversionTo<Schwarzschild<M>> for EddingtonFinkelstein<M> {
    fn convert_point(p: &Point<Self>) -> Point<Schwarzschild<M>> {
        let u = p[0];
        let r = p[1];
        let m = M::mass();
        let t = u - r - (0.5 * (r - 2.0 * m) / m).ln() * 2.0 * m;
        Point::new(arr![f64; t, r, p[2], p[3]])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<Schwarzschild<M>> {
        let r = p[1];
        let m = M::mass();
        let dtdr = -r / (r - 2.0 * m);
        Matrix::new(
            Self::convert_point(p),
            arr![f64;
                        1.0, dtdr, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
            ],
        )
    }

    fn inv_jacobian(
        p: &Point<Self>,
    ) -> Tensor<Schwarzschild<M>, (CovariantIndex, ContravariantIndex)> {
        let r = p[1];
        let m = M::mass();
        let dtdr = -r / (r - 2.0 * m);
        Tensor::<Schwarzschild<M>, (CovariantIndex, ContravariantIndex)>::new(
            Self::convert_point(p),
            arr![f64;
                        1.0, -dtdr, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
            ],
        )
    }
}
