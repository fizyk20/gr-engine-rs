use super::{EddingtonFinkelstein, Mass, NearPole0Schw, NearPolePiSchw};
use diffgeom::coordinates::{ConversionTo, CoordinateSystem, Point};
use diffgeom::metric::MetricSystem;
use diffgeom::tensors::{ContravariantIndex, CovariantIndex, InvTwoForm, Matrix, Tensor, TwoForm};
use std::f64::consts::PI;
use std::marker::PhantomData;
use typenum::consts::U4;

pub struct Schwarzschild<M: Mass> {
    _m: PhantomData<M>,
}

impl<M: Mass> CoordinateSystem for Schwarzschild<M> {
    type Dimension = U4;
}

impl<M: Mass> MetricSystem for Schwarzschild<M> {
    fn g(x: &Point<Self>) -> TwoForm<Self> {
        let r = x[1];
        let th = x[2];
        let m = M::mass();
        let coeff = 1.0 - 2.0 * m / r;
        TwoForm::new(
            x.clone(),
            arr![f64;
                coeff, 0.0, 0.0, 0.0,
                0.0, -1.0/coeff, 0.0, 0.0,
                0.0, 0.0, -r*r, 0.0,
                0.0, 0.0, 0.0, -r*r*th.sin()*th.sin()
            ],
        )
    }

    fn inv_g(x: &Point<Self>) -> InvTwoForm<Self> {
        let r = x[1];
        let th = x[2];
        let m = M::mass();
        let coeff = 1.0 - 2.0 * m / r;
        InvTwoForm::new(
            x.clone(),
            arr![f64;
                1.0 / coeff, 0.0, 0.0, 0.0,
                0.0, -coeff, 0.0, 0.0,
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
                0.0, mr/r2m, 0.0, 0.0,
                mr/r2m, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,

                mr*r2m/r/r, 0.0, 0.0, 0.0,
                0.0, -mr/r2m, 0.0, 0.0,
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

impl<M: Mass + 'static> ConversionTo<EddingtonFinkelstein<M>> for Schwarzschild<M> {
    fn convert_point(p: &Point<Self>) -> Point<EddingtonFinkelstein<M>> {
        let t = p[0];
        let r = p[1];
        let m = M::mass();
        let u = t + r + (0.5 * (r - 2.0 * m) / m).ln() * 2.0 * m;
        Point::new(arr![f64; u, r, p[2], p[3]])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<EddingtonFinkelstein<M>> {
        let r = p[1];
        let m = M::mass();
        let dudr = r / (r - 2.0 * m);
        Matrix::new(
            Self::convert_point(p),
            arr![f64;
                1.0, dudr, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
    }

    fn inv_jacobian(
        p: &Point<Self>,
    ) -> Tensor<EddingtonFinkelstein<M>, (CovariantIndex, ContravariantIndex)> {
        let r = p[1];
        let m = M::mass();
        let dudr = r / (r - 2.0 * m);
        Tensor::<EddingtonFinkelstein<M>, (CovariantIndex, ContravariantIndex)>::new(
            Self::convert_point(p),
            arr![f64;
                1.0, -dudr, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
    }
}

impl<M: Mass + 'static> ConversionTo<NearPole0Schw<M>> for Schwarzschild<M> {
    fn convert_point(p: &Point<Self>) -> Point<NearPole0Schw<M>> {
        let th = p[2];
        let ph = p[3];
        let x = (th / 2.0).tan() * ph.cos();
        let y = (th / 2.0).tan() * ph.sin();
        Point::new(arr![f64; p[0], p[1], x, y])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<NearPole0Schw<M>> {
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
    ) -> Tensor<NearPole0Schw<M>, (CovariantIndex, ContravariantIndex)> {
        let t2 = p[2] / 2.0;
        let ph = p[3];
        let cos2t2 = t2.cos() * t2.cos();
        Tensor::<NearPole0Schw<M>, (CovariantIndex, ContravariantIndex)>::new(
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

impl<M: Mass + 'static> ConversionTo<NearPolePiSchw<M>> for Schwarzschild<M> {
    fn convert_point(p: &Point<Self>) -> Point<NearPolePiSchw<M>> {
        let th = p[2];
        let ph = p[3];
        let x = ((PI - th) / 2.0).tan() * ph.cos();
        let y = ((PI - th) / 2.0).tan() * ph.sin();
        Point::new(arr![f64; p[0], p[1], x, y])
    }

    fn jacobian(p: &Point<Self>) -> Matrix<NearPolePiSchw<M>> {
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
    ) -> Tensor<NearPolePiSchw<M>, (CovariantIndex, ContravariantIndex)> {
        let t2 = p[2] / 2.0;
        let ph = p[3];
        let cos2t2 = t2.cos() * t2.cos();
        Tensor::<NearPolePiSchw<M>, (CovariantIndex, ContravariantIndex)>::new(
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
