extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::string::ToString;

use crate::types::fp::CtFp;
use crate::types::fr::CtFr;
use kzg::common_utils::log_2_byte;
use kzg::eip_4844::BYTES_PER_G1;
use kzg::G1Affine;
use kzg::G1GetFp;
use kzg::G1ProjAddAffine;
use kzg::{G1Mul, G1};

use crate::consts::{G1_GENERATOR, G1_IDENTITY, G1_NEGATIVE_GENERATOR};
// use crate::kzg_proofs::g1_linear_combination;

use constantine_sys as constantine;

use constantine_sys::{
    bls12_381_fp, bls12_381_g1_aff, bls12_381_g1_jac, ctt_bls12_381_g1_jac_cneg_in_place,
    ctt_bls12_381_g1_jac_double, ctt_bls12_381_g1_jac_from_affine, ctt_bls12_381_g1_jac_is_eq,
    ctt_bls12_381_g1_jac_is_inf, ctt_bls12_381_g1_jac_sum,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Eq)]
pub struct CtG1(pub bls12_381_g1_jac);

impl PartialEq for CtG1 {
    fn eq(&self, other: &Self) -> bool {
        unsafe { constantine::ctt_bls12_381_g1_jac_is_eq(&self.0, &other.0) != 0 }
    }
}

impl CtG1 {
    pub(crate) const fn from_xyz(x: bls12_381_fp, y: bls12_381_fp, z: bls12_381_fp) -> Self {
        CtG1(bls12_381_g1_jac { x, y, z })
    }

    pub const fn from_blst_p1(p1: blst::blst_p1) -> Self {
        Self(bls12_381_g1_jac {
            x: bls12_381_fp { limbs: p1.x.l },
            y: bls12_381_fp { limbs: p1.y.l },
            z: bls12_381_fp { limbs: p1.z.l },
        })
    }

    pub const fn to_blst_p1(&self) -> blst::blst_p1 {
        blst::blst_p1 {
            x: blst::blst_fp { l: self.0.x.limbs },
            y: blst::blst_fp { l: self.0.y.limbs },
            z: blst::blst_fp { l: self.0.z.limbs },
        }
    }
}

impl G1 for CtG1 {
    fn identity() -> Self {
        G1_IDENTITY
    }

    fn generator() -> Self {
        G1_GENERATOR
    }

    fn negative_generator() -> Self {
        G1_NEGATIVE_GENERATOR
    }

    #[cfg(feature = "rand")]
    fn rand() -> Self {
        let result: CtG1 = G1_GENERATOR;
        result.mul(&kzg::Fr::rand())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_G1,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_G1]| {
                let mut tmp = bls12_381_g1_aff::default();
                let mut g1 = bls12_381_g1_jac::default();
                unsafe {
                    let tmp_ref: &mut blst::blst_p1_affine = core::mem::transmute(&mut tmp);
                    // The uncompress routine also checks that the point is on the curve
                    if blst::blst_p1_uncompress(tmp_ref, bytes.as_ptr())
                        != blst::BLST_ERROR::BLST_SUCCESS
                    {
                        return Err("Failed to uncompress".to_string());
                    }
                    ctt_bls12_381_g1_jac_from_affine(&mut g1, &tmp);
                }
                Ok(CtG1(g1))
            })
    }

    fn from_hex(hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(&hex[2..]).unwrap();
        Self::from_bytes(&bytes)
    }

    fn to_bytes(&self) -> [u8; 48] {
        let mut out = [0u8; BYTES_PER_G1];
        unsafe {
            let inp_ref: &blst::blst_p1 = core::mem::transmute(&self.0);
            blst::blst_p1_compress(out.as_mut_ptr(), inp_ref);
        }
        out
    }

    fn add_or_dbl(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_sum(&mut ret.0, &self.0, &b.0);
        }
        ret
    }

    fn is_inf(&self) -> bool {
        unsafe { constantine::ctt_bls12_381_g1_jac_is_inf(&self.0) != 0 }
    }

    fn is_valid(&self) -> bool {
        unsafe {
            // FIXME: Constantine equivalent
            blst::blst_p1_in_g1(core::mem::transmute(&self.0))
        }
    }

    fn dbl(&self) -> Self {
        let mut result = bls12_381_g1_jac::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_double(&mut result, &self.0);
        }
        Self(result)
    }

    fn add(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_sum(&mut ret.0, &self.0, &b.0);
        }
        ret
    }

    fn sub(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_diff(&mut ret.0, &self.0, &b.0);
        }
        ret
    }

    fn equals(&self, b: &Self) -> bool {
        unsafe { constantine::ctt_bls12_381_g1_jac_is_eq(&self.0, &b.0) != 0 }
    }

    // FIXME: Wrong here
    const ZERO: Self = CtG1::from_xyz(
        bls12_381_fp { limbs: [0; 6] },
        bls12_381_fp { limbs: [0; 6] },
        bls12_381_fp { limbs: [0; 6] },
    );

    fn add_or_dbl_assign(&mut self, b: &Self) {
        unsafe {
            constantine::ctt_bls12_381_g1_jac_add_in_place(&mut self.0, &b.0);
        }
    }

    fn add_assign(&mut self, b: &Self) {
        unsafe {
            constantine::ctt_bls12_381_g1_jac_add_in_place(&mut self.0, &b.0);
        }
    }

    fn dbl_assign(&mut self) {
        unsafe {
            constantine::ctt_bls12_381_g1_jac_double_in_place(&mut self.0);
        }
    }
}

impl G1Mul<CtFr> for CtG1 {
    fn mul(&self, b: &CtFr) -> Self {
        // FIXME: No transmute here, use constantine
        let mut scalar = blst::blst_scalar::default();
        unsafe {
            blst::blst_scalar_from_fr(&mut scalar, core::mem::transmute(&b.0));
        }

        // Count the number of bytes to be multiplied.
        let mut i = scalar.b.len();
        while i != 0 && scalar.b[i - 1] == 0 {
            i -= 1;
        }

        let mut result = Self::default();
        if i == 0 {
            return G1_IDENTITY;
        } else if i == 1 && scalar.b[0] == 1 {
            return *self;
        } else {
            // Count the number of bits to be multiplied.
            unsafe {
                blst::blst_p1_mult(
                    core::mem::transmute(&mut result.0),
                    core::mem::transmute(&self.0),
                    &(scalar.b[0]),
                    8 * i - 7 + log_2_byte(scalar.b[i - 1]),
                );
            }
        }
        result
    }

    fn g1_lincomb(points: &[Self], scalars: &[CtFr], len: usize) -> Self {
        let mut out = CtG1::default();
        crate::kzg_proofs::g1_linear_combination(&mut out, points, scalars, len);
        out
    }
}

impl G1GetFp<CtFp> for CtG1 {
    fn x(&self) -> &CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&self.0.x)
        }
    }

    fn y(&self) -> &CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&self.0.y)
        }
    }

    fn z(&self) -> &CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&self.0.z)
        }
    }

    fn x_mut(&mut self) -> &mut CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&mut self.0.x)
        }
    }

    fn y_mut(&mut self) -> &mut CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&mut self.0.y)
        }
    }

    fn z_mut(&mut self) -> &mut CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&mut self.0.z)
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct CtG1Affine(pub constantine::bls12_381_g1_aff);

impl G1Affine<CtG1, CtFp> for CtG1Affine {
    const ZERO: Self = Self(bls12_381_g1_aff {
        x: {
            bls12_381_fp {
                limbs: [0, 0, 0, 0, 0, 0],
            }
        },
        y: {
            bls12_381_fp {
                limbs: [0, 0, 0, 0, 0, 0],
            }
        },
    });

    fn into_affine(g1: &CtG1) -> Self {
        let mut ret: Self = Default::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_affine(&mut ret.0, &g1.0);
        }
        ret
    }

    fn into_affines_loc(out: &mut [Self], g1: &[CtG1]) {
        g1.iter()
            .zip(out.iter_mut())
            .for_each(|(g, out_slot)| unsafe {
                constantine::ctt_bls12_381_g1_jac_affine(&mut out_slot.0, &g.0);
            });
    }

    fn into_affines(g1: &[CtG1]) -> Vec<Self> {
        g1.iter()
            .map(|g| {
                let mut ret = Self::default();
                unsafe {
                    constantine::ctt_bls12_381_g1_jac_affine(&mut ret.0, &g.0);
                }
                ret
            })
            .collect::<Vec<_>>()
    }

    fn to_proj(&self) -> CtG1 {
        let mut ret: CtG1 = Default::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_from_affine(&mut ret.0, &self.0);
        }
        ret
    }

    fn x(&self) -> &CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&self.0.x)
        }
    }

    fn y(&self) -> &CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&self.0.y)
        }
    }

    fn is_infinity(&self) -> bool {
        unsafe { constantine::ctt_bls12_381_g1_aff_is_inf(&self.0) != 0 }
    }

    fn x_mut(&mut self) -> &mut CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&mut self.0.x)
        }
    }

    fn y_mut(&mut self) -> &mut CtFp {
        unsafe {
            // Transmute safe due to repr(C) on CtFp
            core::mem::transmute(&mut self.0.y)
        }
    }
}

pub struct CtG1ProjAddAffine;
impl G1ProjAddAffine<CtG1, CtFp, CtG1Affine> for CtG1ProjAddAffine {
    fn add_assign_affine(proj: &mut CtG1, aff: &CtG1Affine) {
        let mut g1_jac = bls12_381_g1_jac::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_from_affine(&mut g1_jac, &aff.0);
            constantine::ctt_bls12_381_g1_jac_add_in_place(&mut proj.0, &g1_jac);
        }
    }

    fn add_or_double_assign_affine(proj: &mut CtG1, aff: &CtG1Affine) {
        let mut g1_jac = bls12_381_g1_jac::default();
        unsafe {
            constantine::ctt_bls12_381_g1_jac_from_affine(&mut g1_jac, &aff.0);
            constantine::ctt_bls12_381_g1_jac_add_in_place(&mut proj.0, &g1_jac);
        }
    }
}