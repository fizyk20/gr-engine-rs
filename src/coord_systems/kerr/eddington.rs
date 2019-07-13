use super::{NearPole0EF, NearPolePiEF, Properties};
use crate::typenum::consts::U4;
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{ContravariantIndex, CovariantIndex, InvTwoForm, Matrix, Tensor, TwoForm};
use generic_array::arr;
use std::f64::consts::PI;
use std::marker::PhantomData;

pub struct EddingtonFinkelstein<P: Properties> {
    _m: PhantomData<P>,
}

impl<P: Properties> CoordinateSystem for EddingtonFinkelstein<P> {
    type Dimension = U4;
}

impl<P: Properties> MetricSystem for EddingtonFinkelstein<P> {
    fn g(x: &Point<Self>) -> TwoForm<Self> {
        let r = x[1];
        let th = x[2];
        let m = P::mass();
        let a = P::ang_momentum();
        let rho2 = r * r + a * a * th.cos() * th.cos();
        let z = 2.0 * m * r * a * th.sin() * th.sin() / rho2;
        TwoForm::new(
            x.clone(),
            arr![f64;
                 1.0 - 2.0*m*r/rho2, -1.0,                  0.0,   z,
                -1.0,                 0.0,                  0.0,   a*th.sin()*th.sin(),
                 0.0,                 0.0,                 -rho2,  0.0,
                 z,                   a*th.sin()*th.sin(),  0.0,  -(r*r + a*a + z)*th.sin()*th.sin()
            ],
        )
    }

    fn inv_g(x: &Point<Self>) -> InvTwoForm<Self> {
        let r = x[1];
        let th = x[2];
        let m = P::mass();
        let a = P::ang_momentum();
        let rho2 = r * r + a * a * th.cos() * th.cos();
        let delta = r * r - 2.0 * m * r + a * a;
        InvTwoForm::new(
            x.clone(),
            arr![f64;
                -a*a*th.sin()*th.sin()/rho2, -(r*r + a*a)/rho2,  0.0,      -a/rho2,
                -(r*r + a*a)/rho2,           -delta/rho2,        0.0,      -a/rho2,
                 0.0,                         0.0,              -1.0/rho2,  0.0,
                -a/rho2,                     -a/rho2,            0.0,      -1.0/(rho2*th.sin()*th.sin())
            ],
        )
    }

    fn christoffel(
        x: &Point<Self>,
    ) -> Tensor<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))> {
        let r = x[1];
        let th = x[2];
        let m = P::mass();
        let a = P::ang_momentum();
        let rho2 = r * r + a * a * th.cos() * th.cos();
        let rho2x = r * r - a * a * th.cos() * th.cos();
        let rho4 = rho2 * rho2;
        let rho6 = rho2 * rho4;
        let delta = r * r - 2.0 * m * r + a * a;

        let uuu = m * (r * r + a * a) * rho2x / rho6;
        let uut = -m * r * a * a * (2.0 * th).sin() / rho4;
        let uup = -m * a * (r * r + a * a) * rho2x * th.sin() * th.sin() / rho6;
        let urt = -a * a * th.sin() * th.cos() / rho2;
        let urp = a * r * th.sin() * th.sin() / rho2;
        let utt = -r * (r * r + a * a) / rho2;
        let utp = 2.0 * m * r * a * a * a * th.sin() * th.sin() * th.sin() * th.cos() / rho4;
        let upp = (r * r + a * a) / rho2
            * (m * rho2x * a * a * th.sin() * th.sin() * th.sin() * th.sin() / rho4
                - r * th.sin() * th.sin());

        let ruu = m * rho2x * delta / rho6;
        let rur = -m * rho2x / rho4;
        let rup = -m * rho2x * delta * a * th.sin() * th.sin() / rho6;
        let rrt = -a * a * th.sin() * th.cos() / rho2;
        let rrp = (r * rho2 + m * rho2x) * a * th.sin() * th.sin() / rho4;
        let rtt = -r * delta / rho2;
        let rpp =
            delta * th.sin() * th.sin() * (m * a * a * rho2x * th.sin() * th.sin() - r * rho4)
                / rho6;

        let tuu = -2.0 * m * r * a * a * th.sin() * th.cos() / rho6;
        let tup = 2.0 * m * r * a * (r * r + a * a) * th.sin() * th.cos() / rho6;
        let trt = r / rho2;
        let trp = a * th.sin() * th.cos() / rho2;
        let ttt = -a * a * th.sin() * th.cos() / rho2;
        let tpp = -th.sin()
            * th.cos()
            * (rho4 * (r * r + a * a)
                + 2.0 * m * r * a * a * th.sin() * th.sin() * (r * r + a * a + rho2))
            / rho6;

        let puu = m * a * rho2x / rho6;
        let put = -2.0 * m * r * a * th.cos() / (rho4 * th.sin());
        let pup = -m * a * a * rho2x * th.sin() * th.sin() / rho6;
        let prt = -a / rho2 * th.cos() / th.sin();
        let prp = r / rho2;
        let ptt = -a * r / rho2;
        let ptp = th.cos() / th.sin() * (1.0 + 2.0 * m * r * a * a * th.sin() * th.sin() / rho4);
        let ppp =
            a * th.sin() * th.sin() * (m * a * a * rho2x * th.sin() * th.sin() - r * rho4) / rho6;

        Tensor::<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))>::new(
            x.clone(),
            arr![f64;
                uuu, 0.0, uut, uup,
                0.0, 0.0, urt, urp,
                uut, urt, utt, utp,
                uup, urp, utp, upp,

                ruu, rur, 0.0, rup,
                rur, 0.0, rrt, rrp,
                0.0, rrt, rtt, 0.0,
                rup, rrp, 0.0, rpp,

                tuu, 0.0, 0.0, tup,
                0.0, 0.0, trt, trp,
                0.0, trt, ttt, 0.0,
                tup, trp, 0.0, tpp,

                puu, 0.0, put, pup,
                0.0, 0.0, prt, prp,
                put, prt, ptt, ptp,
                pup, prp, ptp, ppp
            ],
        )
    }

    // TODO
    //fn dg(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
    //fn covariant_christoffel(x: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>
}

impl<P: Properties + 'static> ConversionTo<NearPole0EF<P>> for EddingtonFinkelstein<P> {
    fn convert_point(p: &Point<Self>) -> Point<NearPole0EF<P>> {
        let th = p[2];
        let ph = p[3];
        let x = (th / 2.0).tan() * ph.cos();
        let y = (th / 2.0).tan() * ph.sin();
        Point::new(arr![f64; p[0], p[1], x, y])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<NearPole0EF<P>> {
        let t2 = p[2] / 2.0;
        let ph = p[3];
        let tan2th1 = 1.0 + t2.tan() * t2.tan();
        Matrix::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 2.0 * ph.cos() / tan2th1, 2.0 * ph.sin() / tan2th1,
                0.0, 0.0, -ph.sin() / t2.tan(), ph.cos() / t2.tan(),
            ],
        )
    }

    fn inv_jacobian(
        p: &Point<Self>,
    ) -> Tensor<NearPole0EF<P>, (CovariantIndex, ContravariantIndex)> {
        let t2 = p[2] / 2.0;
        let ph = p[3];
        let cos2t2 = t2.cos() * t2.cos();
        Tensor::<NearPole0EF<P>, (CovariantIndex, ContravariantIndex)>::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.5 * ph.cos() / cos2t2, -t2.tan() * ph.sin(),
                0.0, 0.0, 0.5 * ph.sin() / cos2t2, t2.tan() * ph.cos(),
            ],
        )
    }
}

impl<P: Properties + 'static> ConversionTo<NearPolePiEF<P>> for EddingtonFinkelstein<P> {
    fn convert_point(p: &Point<Self>) -> Point<NearPolePiEF<P>> {
        let th = p[2];
        let ph = p[3];
        let x = ((PI - th) / 2.0).tan() * ph.cos();
        let y = ((PI - th) / 2.0).tan() * ph.sin();
        Point::new(arr![f64; p[0], p[1], x, y])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<NearPolePiEF<P>> {
        let t2 = p[2] / 2.0;
        let ph = p[3];
        let tan2th1 = 1.0 + t2.tan() * t2.tan();
        Matrix::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -2.0 * ph.cos() / tan2th1, -2.0 * ph.sin() / tan2th1,
                0.0, 0.0, -ph.sin() / t2.tan(), ph.cos() / t2.tan(),
            ],
        )
    }

    fn inv_jacobian(
        p: &Point<Self>,
    ) -> Tensor<NearPolePiEF<P>, (CovariantIndex, ContravariantIndex)> {
        let t2 = p[2] / 2.0;
        let ph = p[3];
        let cos2t2 = t2.cos() * t2.cos();
        Tensor::<NearPolePiEF<P>, (CovariantIndex, ContravariantIndex)>::new(
            Self::convert_point(p),
            arr![f64;
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -0.5 * ph.cos() / cos2t2, -t2.tan() * ph.sin(),
                0.0, 0.0, -0.5 * ph.sin() / cos2t2, t2.tan() * ph.cos(),
            ],
        )
    }
}
