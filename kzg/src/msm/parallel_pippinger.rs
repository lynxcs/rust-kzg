use crate::{
    cfg_into_iter, common_utils::log2_u64, G1Affine, G1Fp, G1ProjAddAffine, Scalar256, G1,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn parallel_pippinger<
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    bases: &[TG1Affine],
    scalars: &[Scalar256],
) -> TG1 {
    const NUM_BITS: u32 = 255;

    // Limit scalars & bases to lower of the two
    let size = std::cmp::min(bases.len(), scalars.len());
    let scalars = &scalars[..size];
    let bases = &bases[..size];

    let scalars_and_bases_iter = scalars
        .iter()
        .zip(bases)
        .filter(|(s, _)| **s != Scalar256::ZERO);
    let c = if size < 32 {
        3
    } else {
        ((log2_u64(size) * 69 / 100) as usize) + 2
    };

    // Divide 0..NUM_BITS into windows of size c & process in parallel
    let mut window_sums = [TG1::ZERO; NUM_BITS as usize];
    cfg_into_iter!(0..NUM_BITS)
        .step_by(c)
        .zip(&mut window_sums)
        .for_each(|(w_start, window_sums)| {
            // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
            let mut buckets = vec![TG1::ZERO; (1 << c) - 1];
            scalars_and_bases_iter.clone().for_each(|(scalar, base)| {
                if *scalar == Scalar256::ONE {
                    if w_start == 0 {
                        ProjAddAffine::add_assign_affine(window_sums, base);
                    }
                } else {
                    let mut scalar = scalar.data;
                    scalar_divn(&mut scalar, w_start);
                    let scalar = scalar[0] % (1 << c);
                    if scalar != 0 {
                        let idx = (scalar - 1) as usize;
                        ProjAddAffine::add_or_double_assign_affine(&mut buckets[idx], base);
                    }
                }
            });

            let mut running_sum = TG1::ZERO;
            buckets.into_iter().rev().for_each(|b| {
                running_sum.add_or_dbl_assign(&b);
                window_sums.add_or_dbl_assign(&running_sum);
            });
        });

    // Traverse windows from high to low
    let lowest = window_sums.first().unwrap();
    lowest.add(
        &window_sums[1..]
            .iter()
            .rev()
            .fold(TG1::ZERO, |mut total, sum_i| {
                total.add_assign(sum_i);
                for _ in 0..c {
                    total.dbl_assign();
                }
                total
            }),
    )
}

fn scalar_divn<const N: usize>(input: &mut [u64; N], mut n: u32) {
    if n >= (64 * N) as u32 {
        *input = [0u64; N];
        return;
    }

    while n >= 64 {
        let mut t = 0;
        for i in 0..N {
            core::mem::swap(&mut t, &mut input[N - i - 1]);
        }
        n -= 64;
    }

    if n > 0 {
        let mut t = 0;
        #[allow(unused)]
        for i in 0..N {
            let a = &mut input[N - i - 1];
            let t2 = *a << (64 - n);
            *a >>= n;
            *a |= t;
            t = t2;
        }
    }
}
