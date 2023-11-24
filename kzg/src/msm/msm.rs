use crate::{
    msm::bucket_msm::BucketMSM,
    msm::glv::decompose,
    msm::types::{G1_SCALAR_SIZE_GLV, GROUP_SIZE_IN_BITS}, Scalar256, G1Affine, G1, G1Fp, G1ProjAddAffine, Fr, common_utils::log2_u64,
};
use std::any::TypeId;

pub struct VariableBaseMSM;

impl VariableBaseMSM {
    /// WARNING: this function is derived from benchmark results running
    /// on a Ubuntu 20.04.2 LTS server with AMD EPYC 7282 16-Core CPU
    /// and 128G memory, the optimal performance may vary on a different
    /// configuration.
    const fn get_opt_window_size(k: u32) -> u32 {
        match k {
            0..=9 => 8,
            10..=12 => 10,
            13..=14 => 12,
            15..=19 => 13,
            20..=22 => 15,
            23.. => 16,
        }
    }

    fn msm_slice(mut scalar: Scalar256, slices: &mut [u32], window_bits: u32) {
        assert!(window_bits <= 31); // reserve one bit for marking signed slices

        let mut carry = 0;
        let total = 1 << window_bits;
        let half = total >> 1;
        slices.iter_mut().for_each(|el| {
            *el = (scalar.data[0] % (1 << window_bits)) as u32;
            scalar.divn(window_bits);

            *el += carry;
            if half < *el {
                // slices[i] == half is okay, since (slice[i]-1) will be used for bucket_id
                *el = total - *el;
                carry = 1;
                *el |= 1 << 31; // mark the highest bit for later
            } else {
                carry = 0;
            }
        });
        assert!(
            carry == 0,
            "msm_slice overflows when apply signed-bucket-index"
        );
    }

    fn multi_scalar_mul_g1_glv<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>, TProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>, TFr: Fr>(
        points: &[TG1Affine],
        scalars: &[Scalar256],
        window_bits: u32,
        max_batch: u32,
        max_collisions: u32,
    ) -> TG1 {
        let num_slices: usize = ((G1_SCALAR_SIZE_GLV + window_bits - 1) / window_bits) as usize;
        let mut bucket_msm =
            BucketMSM::<TG1, TG1Fp, TG1Affine, TProjAddAffine>::new(G1_SCALAR_SIZE_GLV, window_bits, max_batch, max_collisions);
        // scalar = phi * lambda + normal
        let mut phi_slices = vec![0u32; num_slices];
        let mut normal_slices = vec![0u32; num_slices];

        scalars
            .iter()
            .zip(points)
            .filter(|(s, _)| !s.is_zero())
            .for_each(|(scalar, point)| {
                // TODO
                // use unsafe cast for type conversion until we have a better approach
                // let g1_scalar: G1BigInt =
                //     unsafe { *(std::ptr::addr_of!(scalar) as *const G1BigInt) };

                let (phi, normal, is_neg_scalar, is_neg_normal) =
                    decompose(&TFr::from_u64_arr(&scalar.data), window_bits);

                Self::msm_slice(
                    Scalar256::from_u64(phi.to_u64_arr()),
                    &mut phi_slices[..num_slices],
                    window_bits,
                );
                Self::msm_slice(
                    Scalar256::from_u64(phi.to_u64_arr()),
                    &mut normal_slices[..num_slices],
                    window_bits,
                );
                bucket_msm.process_point_and_slices_glv(
                    point,
                    &normal_slices[..num_slices],
                    &phi_slices[..num_slices],
                    is_neg_scalar,
                    is_neg_normal,
                );
            });

        bucket_msm.process_complete();
        bucket_msm.batch_reduce()
    }

    // fn multi_scalar_mul_general<P: Parameters>(
    //     points: &[Affine<P>],
    //     scalars: &[BigInt<P>],
    //     window_bits: u32,
    //     max_batch: u32,
    //     max_collisions: u32,
    // ) -> Projective<P> {
    //     let scalar_size = <P::ScalarField as PrimeField>::MODULUS_BIT_SIZE as u32;
    //     let num_slices: usize = ((scalar_size + window_bits - 1) / window_bits) as usize;
    //     let mut bucket_msm =
    //         BucketMSM::<P>::new(scalar_size, window_bits, max_batch, max_collisions);

    //     let mut slices = vec![0u32; num_slices];
    //     scalars
    //         .iter()
    //         .zip(points)
    //         .filter(|(s, _)| !s.is_zero())
    //         .for_each(|(&scalar, point)| {
    //             Self::msm_slice::<P>(scalar, &mut slices[..num_slices], window_bits);
    //             bucket_msm.process_point_and_slices(point, &slices[..num_slices]);
    //         });

    //     bucket_msm.process_complete();
    //     bucket_msm.batch_reduce()
    // }

    // pub fn multi_scalar_mul_custom<P: Parameters>(
    //     points: &[Affine<P>],
    //     scalars: &[BigInt<P>],
    //     window_bits: u32,
    //     max_batch: u32,
    //     max_collisions: u32,
    //     glv_enabled: bool,
    // ) -> Projective<P> {
    //     assert!(
    //         window_bits as usize > GROUP_SIZE_IN_BITS,
    //         "Window_bits must be greater than the default log(group size)"
    //     );
    //     if glv_enabled {
    //         assert!(
    //             TypeId::of::<P>() == TypeId::of::<G1Parameters>(),
    //             "Glv is only supported for bls12-381 curve!"
    //         );
    //         Self::multi_scalar_mul_g1_glv(points, scalars, window_bits, max_batch, max_collisions)
    //     } else {
    //         Self::multi_scalar_mul_general(points, scalars, window_bits, max_batch, max_collisions)
    //     }
    // }

    pub fn multi_scalar_mul<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>, TProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>, TFr: Fr>(
        points: &[TG1Affine],
        scalars: &[Scalar256],
    ) -> TG1 {
        let opt_window_size = Self::get_opt_window_size(log2_u64(points.len()) as u32);
        Self::multi_scalar_mul_g1_glv::<TG1, TG1Fp, TG1Affine, TProjAddAffine, TFr>(points, scalars, opt_window_size, 2048, 256)
    }
}

// #[cfg(test)]
// mod collision_method_pippenger_tests {
//     use super::*;
//     use ark_bls12_381::g1::{Config, self};

//     #[test]
//     fn test_msm_slice_window_size_1() {
//         let scalar = G1BigInt::from(0b101 as u32);
//         let mut slices: Vec<u32> = vec![0; 3];
//         VariableBaseMSM::msm_slice::<g1::Config>(scalar, &mut slices, 1);
//         // print!("slices {:?}\n", slices);
//         assert!(slices.iter().eq([1, 0, 1].iter()));
//     }
//     #[test]
//     fn test_msm_slice_window_size_2() {
//         let scalar = G1BigInt::from(0b000110 as u32);
//         let mut slices: Vec<u32> = vec![0; 3];
//         VariableBaseMSM::msm_slice::<g1::Config>(scalar, &mut slices, 2);
//         assert!(slices.iter().eq([2, 1, 0].iter()));
//     }

//     #[test]
//     fn test_msm_slice_window_size_3() {
//         let scalar = G1BigInt::from(0b010111000 as u32);
//         let mut slices: Vec<u32> = vec![0; 3];
//         VariableBaseMSM::msm_slice::<g1::Config>(scalar, &mut slices, 3);
//         assert!(slices.iter().eq([0, 0x80000001, 3].iter()));
//     }

//     #[test]
//     fn test_msm_slice_window_size_16() {
//         let scalar = G1BigInt::from(0x123400007FFF as u64);
//         let mut slices: Vec<u32> = vec![0; 3];
//         VariableBaseMSM::msm_slice::<g1::Config>(scalar, &mut slices, 16);
//         assert!(slices.iter().eq([0x7FFF, 0, 0x1234].iter()));
//     }
// }
