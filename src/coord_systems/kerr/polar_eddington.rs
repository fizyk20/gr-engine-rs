use super::{EddingtonFinkelstein, Properties};
use crate::typenum::consts::U4;
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{ContravariantIndex, CovariantIndex, InvTwoForm, Matrix, Tensor, TwoForm};
use generic_array::arr;
use std::f64::consts::PI;
use std::marker::PhantomData;

/// The coordinate system near the pole theta=pi
pub struct NearPole0EF<P: Properties> {
    _m: PhantomData<P>,
}

impl<P: Properties> CoordinateSystem for NearPole0EF<P> {
    type Dimension = U4;
}

impl<P: Properties> MetricSystem for NearPole0EF<P> {
    fn g(p: &Point<Self>) -> TwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = P::mass();
        let a = P::ang_momentum();

        let xy1 = 1.0 + x * x + y * y;
        let negxy1 = 1.0 - x * x - y * y;
        let rho2 = r * r + a * a * negxy1 * negxy1 / xy1 / xy1;
        let alpha2 = 4.0 / xy1 / xy1;

        let guu = 1.0 - 2.0 * m * r / rho2;
        let gxx = -alpha2 * (r * r + a * a - alpha2 * a * a * (x * x - 2.0 * m * r * y * y / rho2));
        let gyy = -alpha2 * (r * r + a * a - alpha2 * a * a * (y * y - 2.0 * m * r * x * x / rho2));
        let gux = -2.0 * m * r * a * y * alpha2 / rho2;
        let guy = 2.0 * m * r * a * x * alpha2 / rho2;
        let grx = -alpha2 * a * y;
        let gry = alpha2 * a * x;
        let gxy = x * y * a * a * alpha2 * alpha2 * (1.0 + 2.0 * m * r / rho2);
        TwoForm::new(
            p.clone(),
            arr![f64;
                guu, -1.0, gux, guy,
                -1.0, 0.0, grx, gry,
                gux, grx, gxx, gxy,
                guy, gry, gxy, gyy
            ],
        )
    }

    fn inv_g(p: &Point<Self>) -> InvTwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = P::mass();
        let a = P::ang_momentum();

        let xy1 = 1.0 + x * x + y * y;
        let negxy1 = 1.0 - x * x - y * y;
        let rho2 = r * r + a * a * negxy1 * negxy1 / xy1 / xy1;
        let alpha2 = 4.0 / xy1 / xy1;

        let guu = -alpha2 * a * a * (x * x + y * y) / rho2;
        let grr = -(r * r + a * a - 2.0 * m * r) / rho2;
        let gxx = -1.0 / alpha2 / rho2;
        let gyy = gxx;
        let gur = -(r * r + a * a) / rho2;
        let gux = a * y / rho2;
        let guy = -a * x / rho2;
        let grx = a * y / rho2;
        let gry = -a * x / rho2;
        InvTwoForm::new(
            p.clone(),
            arr![f64;
                guu, gur, gux, guy,
                gur, grr, grx, gry,
                gux, grx, gxx, 0.0,
                guy, gry, 0.0, gyy
            ],
        )
    }
}

// Conversions

impl<P: Properties + 'static> ConversionTo<EddingtonFinkelstein<P>> for NearPole0EF<P> {
    fn convert_point(p: &Point<Self>) -> Point<EddingtonFinkelstein<P>> {
        let x = p[2];
        let y = p[3];
        let th = (x * x + y * y).sqrt().atan() * 2.0;
        let ph = y.atan2(x);
        Point::new(arr![f64; p[0], p[1], th, ph])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<EddingtonFinkelstein<P>> {
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
    ) -> Tensor<EddingtonFinkelstein<P>, (CovariantIndex, ContravariantIndex)> {
        let x = p[2];
        let y = p[3];
        let coeff = 2.0 / (1.0 + x * x + y * y) / (x * x + y * y).sqrt();
        Tensor::<EddingtonFinkelstein<P>, (CovariantIndex, ContravariantIndex)>::new(
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
pub struct NearPolePiEF<P: Properties> {
    _m: PhantomData<P>,
}

impl<P: Properties> CoordinateSystem for NearPolePiEF<P> {
    type Dimension = U4;
}

impl<P: Properties> MetricSystem for NearPolePiEF<P> {
    fn g(p: &Point<Self>) -> TwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = P::mass();
        let a = P::ang_momentum();

        let xy1 = 1.0 + x * x + y * y;
        let negxy1 = 1.0 - x * x - y * y;
        let rho2 = r * r + a * a * negxy1 * negxy1 / xy1 / xy1;
        let alpha2 = 4.0 / xy1 / xy1;

        let guu = 1.0 - 2.0 * m * r / rho2;
        let gxx = -alpha2 * (r * r + a * a - alpha2 * a * a * (x * x - 2.0 * m * r * y * y / rho2));
        let gyy = -alpha2 * (r * r + a * a - alpha2 * a * a * (y * y - 2.0 * m * r * x * x / rho2));
        let gux = -2.0 * m * r * a * y * alpha2 / rho2;
        let guy = 2.0 * m * r * a * x * alpha2 / rho2;
        let grx = -alpha2 * a * y;
        let gry = alpha2 * a * x;
        let gxy = x * y * a * a * alpha2 * alpha2 * (1.0 + 2.0 * m * r / rho2);
        TwoForm::new(
            p.clone(),
            arr![f64;
                guu, -1.0, gux, guy,
                -1.0, 0.0, grx, gry,
                gux, grx, gxx, gxy,
                guy, gry, gxy, gyy
            ],
        )
    }

    fn inv_g(p: &Point<Self>) -> InvTwoForm<Self> {
        let r = p[1];
        let x = p[2];
        let y = p[3];
        let m = P::mass();
        let a = P::ang_momentum();

        let xy1 = 1.0 + x * x + y * y;
        let negxy1 = 1.0 - x * x - y * y;
        let rho2 = r * r + a * a * negxy1 * negxy1 / xy1 / xy1;
        let alpha2 = 4.0 / xy1 / xy1;

        let guu = -alpha2 * a * a * (x * x + y * y) / rho2;
        let grr = -(r * r + a * a - 2.0 * m * r) / rho2;
        let gxx = -1.0 / alpha2 / rho2;
        let gyy = gxx;
        let gur = -(r * r + a * a) / rho2;
        let gux = a * y / rho2;
        let guy = -a * x / rho2;
        let grx = a * y / rho2;
        let gry = -a * x / rho2;
        InvTwoForm::new(
            p.clone(),
            arr![f64;
                guu, gur, gux, guy,
                gur, grr, grx, gry,
                gux, grx, gxx, 0.0,
                guy, gry, 0.0, gyy
            ],
        )
    }
}

// Conversions

impl<P: Properties + 'static> ConversionTo<EddingtonFinkelstein<P>> for NearPolePiEF<P> {
    fn convert_point(p: &Point<Self>) -> Point<EddingtonFinkelstein<P>> {
        let x = p[2];
        let y = p[3];
        let th = PI - (x * x + y * y).sqrt().atan() * 2.0;
        let ph = y.atan2(x);
        Point::new(arr![f64; p[0], p[1], th, ph])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<EddingtonFinkelstein<P>> {
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
    ) -> Tensor<EddingtonFinkelstein<P>, (CovariantIndex, ContravariantIndex)> {
        let x = p[2];
        let y = p[3];
        let coeff = 2.0 / (1.0 + x * x + y * y) / (x * x + y * y).sqrt();
        Tensor::<EddingtonFinkelstein<P>, (CovariantIndex, ContravariantIndex)>::new(
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
