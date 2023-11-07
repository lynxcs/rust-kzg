#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

pub mod common_utils;
pub mod eip_4844;
pub mod msm;

pub trait Fr: Default + Clone + PartialEq + Sync {
    fn null() -> Self;

    fn zero() -> Self;

    fn one() -> Self;

    #[cfg(feature = "rand")]
    fn rand() -> Self;

    fn from_bytes(bytes: &[u8]) -> Result<Self, String>;

    fn from_bytes_unchecked(bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes(bytes)
    }

    fn from_hex(hex: &str) -> Result<Self, String>;

    fn from_u64_arr(u: &[u64; 4]) -> Self;

    fn from_u64(u: u64) -> Self;

    fn to_bytes(&self) -> [u8; 32];

    fn to_u64_arr(&self) -> [u64; 4];

    fn is_one(&self) -> bool;

    fn is_zero(&self) -> bool;

    fn is_null(&self) -> bool;

    fn sqr(&self) -> Self;

    fn mul(&self, b: &Self) -> Self;

    fn add(&self, b: &Self) -> Self;

    fn sub(&self, b: &Self) -> Self;

    fn eucl_inverse(&self) -> Self;

    fn negate(&self) -> Self;

    fn inverse(&self) -> Self;

    fn pow(&self, n: usize) -> Self;

    fn div(&self, b: &Self) -> Result<Self, String>;

    fn equals(&self, b: &Self) -> bool;

    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

pub trait G1: Clone + Default + PartialEq + Sync + Debug + Send {
    const ZERO: Self;

    fn identity() -> Self;

    fn generator() -> Self;

    fn negative_generator() -> Self;

    #[cfg(feature = "rand")]
    fn rand() -> Self;

    fn from_bytes(bytes: &[u8]) -> Result<Self, String>;

    fn from_hex(hex: &str) -> Result<Self, String>;

    fn to_bytes(&self) -> [u8; 48];

    fn add_or_dbl(&mut self, b: &Self) -> Self;

    fn is_inf(&self) -> bool;

    fn is_valid(&self) -> bool;

    fn dbl(&self) -> Self;

    fn add(&self, b: &Self) -> Self;

    fn sub(&self, b: &Self) -> Self;

    fn equals(&self, b: &Self) -> bool;

    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }

    fn add_or_dbl_assign(&mut self, b: &Self);
    fn add_assign(&mut self, b: &Self);
    fn dbl_assign(&mut self);
}

pub trait G1Mul<TFr: Fr>: G1 + Clone {
    fn mul(&self, b: &TFr) -> Self;

    // Instead of creating separate trait, keep linear comb here for simplicity
    fn g1_lincomb(points: &[Self], scalars: &[TFr], len: usize) -> Self;
}

pub trait G1Fp: Clone + Default + Sync + Copy + PartialEq + Debug {
    const ZERO: Self;
    const ONE: Self;

    fn inverse(&self) -> Option<Self>;

    fn square(&self) -> Self;
    fn double(&self) -> Self;

    fn from_underlying_arr(arr: &[u64; 6]) -> Self;

    fn neg_assign(&mut self);

    fn mul_assign_fp(&mut self, b: &Self);

    fn sub_assign_fp(&mut self, b: &Self);

    fn add_assign_fp(&mut self, b: &Self);

    fn neg(mut self) -> Self {
        self.neg_assign();
        self
    }

    fn mul_fp(mut self, b: &Self) -> Self {
        self.mul_assign_fp(b);
        self
    }

    fn sub_fp(mut self, b: &Self) -> Self {
        self.sub_assign_fp(b);
        self
    }

    fn add_fp(mut self, b: &Self) -> Self {
        self.add_assign_fp(b);
        self
    }

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn set_zero(&mut self) {
        *self = Self::ZERO;
    }

    fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    fn set_one(&mut self) {
        *self = Self::ONE;
    }
}

pub trait G1Affine<TG1: G1, TG1Fp: G1Fp>:
    Clone + Default + PartialEq + Sync + Copy + Debug
{
    const ZERO: Self;

    fn into_affine(g1: &TG1) -> Self;
    // Batch conversion can be faster than transforming each individually
    fn into_affines(g1: &[TG1]) -> Vec<Self>;

    fn to_proj(&self) -> TG1;

    // Return field X of Affine
    fn x(&self) -> &TG1Fp;

    // Return field Y of Affine
    fn y(&self) -> &TG1Fp;

    // Return field X of Affine as mutable
    fn x_mut(&mut self) -> &mut TG1Fp;

    // Return field Y of Affine as mutable
    fn y_mut(&mut self) -> &mut TG1Fp;

    // Return whether Affine is at infinity
    fn is_infinity(&self) -> bool;

    // Return whether Affine is zero
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn set_zero(&mut self) {
        *self = Self::ZERO;
    }
}

pub trait G1ProjAddAffine<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>: Sized {
    fn add_assign_affine(proj: &mut TG1, aff: &TG1Affine);

    fn add_or_double_assign_affine(proj: &mut TG1, aff: &TG1Affine);

    fn add_affine(mut proj: TG1, aff: &TG1Affine) -> TG1 {
        Self::add_assign_affine(&mut proj, aff);
        proj
    }

    fn add_or_double_affine(mut proj: TG1, aff: &TG1Affine) -> TG1 {
        Self::add_or_double_assign_affine(&mut proj, aff);
        proj
    }

    fn sub_assign_affine(proj: &mut TG1, mut aff: TG1Affine) {
        aff.y_mut().neg_assign();
        Self::add_assign_affine(proj, &aff);
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Scalar256 {
    data: [u64; 4],
}

#[allow(unused)]
impl Scalar256 {
    const ONE: Self = Self { data: [1, 0, 0, 0] };
    const ZERO: Self = Self { data: [0; 4] };

    fn from_u64(arr: [u64; 4]) -> Self {
        Scalar256 { data: arr }
    }

    pub fn from_u8(arr: &[u8; 32]) -> Self {
        Scalar256 {
            data: Self::cast_scalar_to_u64_arr(arr),
        }
    }

    const fn cast_scalar_to_u64_arr<const N: usize, const N_U8: usize>(
        input: &[u8; N_U8],
    ) -> [u64; N] {
        let ptr = input.as_ptr();

        unsafe { core::slice::from_raw_parts(&*(ptr as *const [u64; N]), 1)[0] }
    }
}

pub trait G2: Clone + Default {
    fn generator() -> Self;

    fn negative_generator() -> Self;

    fn from_bytes(bytes: &[u8]) -> Result<Self, String>;

    fn to_bytes(&self) -> [u8; 96];

    fn add_or_dbl(&mut self, b: &Self) -> Self;

    fn dbl(&self) -> Self;

    fn sub(&self, b: &Self) -> Self;

    fn equals(&self, b: &Self) -> bool;
}

pub trait G2Mul<Fr>: Clone {
    fn mul(&self, b: &Fr) -> Self;
}

pub trait PairingVerify<TG1: G1, TG2: G2> {
    fn verify(a1: &TG1, a2: &TG2, b1: &TG1, b2: &TG2) -> bool;
}

pub trait FFTFr<Coeff: Fr> {
    fn fft_fr(&self, data: &[Coeff], inverse: bool) -> Result<Vec<Coeff>, String>;
}

pub trait FFTG1<Coeff: G1> {
    fn fft_g1(&self, data: &[Coeff], inverse: bool) -> Result<Vec<Coeff>, String>;
}

pub trait DAS<Coeff: Fr> {
    fn das_fft_extension(&self, evens: &[Coeff]) -> Result<Vec<Coeff>, String>;
}

pub trait ZeroPoly<Coeff: Fr, Polynomial: Poly<Coeff>> {
    /// Calculates the minimal polynomial that evaluates to zero for powers of roots of unity at the
    /// given indices.
    /// The returned polynomial has a length of `idxs.len() + 1`.
    ///
    /// Uses straightforward long multiplication to calculate the product of `(x - r^i)` where `r`
    /// is a root of unity and the `i`s are the indices at which it must evaluate to zero.
    fn do_zero_poly_mul_partial(&self, idxs: &[usize], stride: usize)
        -> Result<Polynomial, String>;

    /// Reduce partials using a specified domain size.
    /// Calculates the product of all polynomials via FFT and then applies an inverse FFT to produce
    /// a new Polynomial.
    fn reduce_partials(
        &self,
        domain_size: usize,
        partials: &[Polynomial],
    ) -> Result<Polynomial, String>;

    /// Calculate the minimal polynomial that evaluates to zero for powers of roots of unity that
    /// correspond to missing indices.
    /// This is done simply by multiplying together `(x - r^i)` for all the `i` that are missing
    /// indices, using a combination of direct multiplication ([`Self::do_zero_poly_mul_partial()`])
    /// and iterated multiplication via convolution (#reduce_partials).
    /// Also calculates the FFT (the "evaluation polynomial").
    fn zero_poly_via_multiplication(
        &self,
        domain_size: usize,
        idxs: &[usize],
    ) -> Result<(Vec<Coeff>, Polynomial), String>;
}

pub trait FFTSettings<Coeff: Fr>: Default + Clone {
    fn new(scale: usize) -> Result<Self, String>;

    fn get_max_width(&self) -> usize;

    fn get_expanded_roots_of_unity_at(&self, i: usize) -> Coeff;

    fn get_expanded_roots_of_unity(&self) -> &[Coeff];

    fn get_reverse_roots_of_unity_at(&self, i: usize) -> Coeff;

    fn get_reversed_roots_of_unity(&self) -> &[Coeff];

    fn get_roots_of_unity_at(&self, i: usize) -> Coeff;

    fn get_roots_of_unity(&self) -> &[Coeff];
}

pub trait FFTSettingsPoly<Coeff: Fr, Polynomial: Poly<Coeff>, FSettings: FFTSettings<Coeff>> {
    fn poly_mul_fft(
        a: &Polynomial,
        b: &Polynomial,
        len: usize,
        fs: Option<&FSettings>,
    ) -> Result<Polynomial, String>;
}

pub trait Poly<Coeff: Fr>: Default + Clone {
    fn new(size: usize) -> Self;

    // Default implementation not as efficient, should be implemented by type itself!
    fn from_coeffs(coeffs: &[Coeff]) -> Self {
        let mut poly = Self::new(coeffs.len());

        for (i, coeff) in coeffs.iter().enumerate() {
            poly.set_coeff_at(i, coeff);
        }

        poly
    }

    fn get_coeff_at(&self, i: usize) -> Coeff;

    fn set_coeff_at(&mut self, i: usize, x: &Coeff);

    fn get_coeffs(&self) -> &[Coeff];

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn eval(&self, x: &Coeff) -> Coeff;

    fn scale(&mut self);

    fn unscale(&mut self);

    fn inverse(&mut self, new_len: usize) -> Result<Self, String>;

    fn div(&mut self, x: &Self) -> Result<Self, String>;

    fn long_div(&mut self, x: &Self) -> Result<Self, String>;

    fn fast_div(&mut self, x: &Self) -> Result<Self, String>;

    fn mul_direct(&mut self, x: &Self, len: usize) -> Result<Self, String>;
}

pub trait PolyRecover<Coeff: Fr, Polynomial: Poly<Coeff>, FSettings: FFTSettings<Coeff>> {
    fn recover_poly_coeffs_from_samples(
        samples: &[Option<Coeff>],
        fs: &FSettings,
    ) -> Result<Polynomial, String>;

    fn recover_poly_from_samples(
        samples: &[Option<Coeff>],
        fs: &FSettings,
    ) -> Result<Polynomial, String>;
}

pub trait KZGSettings<
    Coeff1: Fr,
    Coeff2: G1,
    Coeff3: G2,
    Fs: FFTSettings<Coeff1>,
    Polynomial: Poly<Coeff1>,
>: Default + Clone
{
    fn new(
        secret_g1: &[Coeff2],
        secret_g2: &[Coeff3],
        length: usize,
        fs: &Fs,
    ) -> Result<Self, String>;

    fn commit_to_poly(&self, p: &Polynomial) -> Result<Coeff2, String>;

    fn compute_proof_single(&self, p: &Polynomial, x: &Coeff1) -> Result<Coeff2, String>;

    fn check_proof_single(
        &self,
        com: &Coeff2,
        proof: &Coeff2,
        x: &Coeff1,
        value: &Coeff1,
    ) -> Result<bool, String>;

    fn compute_proof_multi(&self, p: &Polynomial, x: &Coeff1, n: usize) -> Result<Coeff2, String>;

    fn check_proof_multi(
        &self,
        com: &Coeff2,
        proof: &Coeff2,
        x: &Coeff1,
        values: &[Coeff1],
        n: usize,
    ) -> Result<bool, String>;

    fn get_expanded_roots_of_unity_at(&self, i: usize) -> Coeff1;

    fn get_roots_of_unity_at(&self, i: usize) -> Coeff1;

    fn get_fft_settings(&self) -> &Fs;

    fn get_g1_secret(&self) -> &[Coeff2];

    fn get_g2_secret(&self) -> &[Coeff3];
}

pub trait FK20SingleSettings<
    Coeff1: Fr,
    Coeff2: G1,
    Coeff3: G2,
    Fs: FFTSettings<Coeff1>,
    Polynomial: Poly<Coeff1>,
    Ks: KZGSettings<Coeff1, Coeff2, Coeff3, Fs, Polynomial>,
>: Default + Clone
{
    fn new(ks: &Ks, n2: usize) -> Result<Self, String>;

    fn data_availability(&self, p: &Polynomial) -> Result<Vec<Coeff2>, String>;

    fn data_availability_optimized(&self, p: &Polynomial) -> Result<Vec<Coeff2>, String>;
}

pub trait FK20MultiSettings<
    Coeff1: Fr,
    Coeff2: G1,
    Coeff3: G2,
    Fs: FFTSettings<Coeff1>,
    Polynomial: Poly<Coeff1>,
    Ks: KZGSettings<Coeff1, Coeff2, Coeff3, Fs, Polynomial>,
>: Default + Clone
{
    fn new(ks: &Ks, n2: usize, chunk_len: usize) -> Result<Self, String>;

    fn data_availability(&self, p: &Polynomial) -> Result<Vec<Coeff2>, String>;

    fn data_availability_optimized(&self, p: &Polynomial) -> Result<Vec<Coeff2>, String>;
}
