use super::{EddingtonFinkelstein, Mass};
use crate::typenum::consts::U4;
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{ContravariantIndex, CovariantIndex, InvTwoForm, Matrix, Tensor, TwoForm};
use generic_array::arr;
use std::f64::consts::PI;
use std::marker::PhantomData;

/// The coordinate system near the pole theta=pi
pub struct NearPole0EF<M: Mass> {
    _m: PhantomData<M>,
}

impl<M: Mass> CoordinateSystem for NearPole0EF<M> {
    type Dimension = U4;
}

impl<M: Mass> MetricSystem for NearPole0EF<M> {
    fn g(p: &Point<Self>) -> TwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = M::mass();
        let xy1 = 1.0 + x * x + y * y;
        let alpha2 = 4.0 / xy1 / xy1;
        TwoForm::new(
            p.clone(),
            arr![f64;
                1.0 - 2.0*m/r, -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -alpha2*r*r, 0.0,
                0.0, 0.0, 0.0, -alpha2*r*r
            ],
        )
    }

    fn inv_g(p: &Point<Self>) -> InvTwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = M::mass();
        let xy1 = 1.0 + x * x + y * y;
        let alpha2 = 4.0 / xy1 / xy1;
        InvTwoForm::new(
            p.clone(),
            arr![f64;
                -1.0 + 2.0*m/r, -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.0/alpha2/r/r, 0.0,
                0.0, 0.0, 0.0, -1.0/alpha2/r/r
            ],
        )
    }
}

// Conversions

impl<M: Mass + 'static> ConversionTo<EddingtonFinkelstein<M>> for NearPole0EF<M> {
    fn convert_point(p: &Point<Self>) -> Point<EddingtonFinkelstein<M>> {
        let x = p[2];
        let y = p[3];
        let th = (x * x + y * y).sqrt().atan() * 2.0;
        let ph = y.atan2(x);
        Point::new(arr![f64; p[0], p[1], th, ph])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<EddingtonFinkelstein<M>> {
        let x = p[2];
        let y = p[3];
        let coeff = (1.0 + x * x + y * y) / 2.0 / (x * x + y * y).sqrt();
        Matrix::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, x * coeff, -y,
                0.0, 0.0, y * coeff, x,
            ],
        )
    }

    fn inv_jacobian(
        p: &Point<Self>,
    ) -> Tensor<EddingtonFinkelstein<M>, (CovariantIndex, ContravariantIndex)> {
        let x = p[2];
        let y = p[3];
        let coeff = 2.0 / (1.0 + x * x + y * y) / (x * x + y * y).sqrt();
        Tensor::<EddingtonFinkelstein<M>, (CovariantIndex, ContravariantIndex)>::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, x * coeff, y * coeff,
                0.0, 0.0, -y/(x*x + y*y), x/(x*x + y*y),
            ],
        )
    }
}

/// The coordinate system near the pole theta=pi
pub struct NearPolePiEF<M: Mass> {
    _m: PhantomData<M>,
}

impl<M: Mass> CoordinateSystem for NearPolePiEF<M> {
    type Dimension = U4;
}

impl<M: Mass> MetricSystem for NearPolePiEF<M> {
    fn g(p: &Point<Self>) -> TwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = M::mass();
        let xy1 = 1.0 + x * x + y * y;
        let alpha2 = 4.0 / xy1 / xy1;
        TwoForm::new(
            p.clone(),
            arr![f64;
                1.0 - 2.0*m/r, -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -alpha2*r*r, 0.0,
                0.0, 0.0, 0.0, -alpha2*r*r
            ],
        )
    }

    fn inv_g(p: &Point<Self>) -> InvTwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = M::mass();
        let xy1 = 1.0 + x * x + y * y;
        let alpha2 = 4.0 / xy1 / xy1;
        InvTwoForm::new(
            p.clone(),
            arr![f64;
                -1.0 + 2.0*m/r, -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.0/alpha2/r/r, 0.0,
                0.0, 0.0, 0.0, -1.0/alpha2/r/r
            ],
        )
    }
}

// Conversions

impl<M: Mass + 'static> ConversionTo<EddingtonFinkelstein<M>> for NearPolePiEF<M> {
    fn convert_point(p: &Point<Self>) -> Point<EddingtonFinkelstein<M>> {
        let x = p[2];
        let y = p[3];
        let th = PI - (x * x + y * y).sqrt().atan() * 2.0;
        let ph = y.atan2(x);
        Point::new(arr![f64; p[0], p[1], th, ph])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<EddingtonFinkelstein<M>> {
        let x = p[2];
        let y = p[3];
        let coeff = (1.0 + x * x + y * y) / 2.0 / (x * x + y * y).sqrt();
        Matrix::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -x * coeff, -y,
                0.0, 0.0, -y * coeff, x,
            ],
        )
    }

    fn inv_jacobian(
        p: &Point<Self>,
    ) -> Tensor<EddingtonFinkelstein<M>, (CovariantIndex, ContravariantIndex)> {
        let x = p[2];
        let y = p[3];
        let coeff = 2.0 / (1.0 + x * x + y * y) / (x * x + y * y).sqrt();
        Tensor::<EddingtonFinkelstein<M>, (CovariantIndex, ContravariantIndex)>::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -x * coeff, -y * coeff,
                0.0, 0.0, -y/(x*x + y*y), x/(x*x + y*y),
            ],
        )
    }
}
