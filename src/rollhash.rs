use const_primes::is_prime;

// Chose a prime number greater than 256 that minimizes hash collisions
// for the prefixes of all mergeable ranks.
// Modulus * prime must be less than 2^63-1 to avoid overflow.
const PRIME: i64 = 997;
const PRIME_INVERSE: i64 = 617853560682069;
const MODULUS: i64 = 1e15 as i64 + 37;

const _: () = assert!(PRIME > 256, "PRIME must be greater than 256 for byte-wise rolling hash");
const _: () = assert!(PRIME < MODULUS, "PRIME must be less than MODULUS");
const _: () = assert!(
    MODULUS as i128 * PRIME as i128 <= i64::MAX as i128,
    "MODULUS * PRIME must not exceed i64::MAX to avoid overflow"
);
const _: () = assert!(
    (PRIME as i128 * PRIME_INVERSE as i128) % MODULUS as i128 == 1,
    "PRIME_INVERSE must be the modular multiplicative inverse of PRIME"
);
const _: () = assert!(is_prime(PRIME as u64), "PRIME must be a prime number");
const _: () = assert!(is_prime(MODULUS as u64), "MODULUS must be a prime number");

#[inline(always)]
pub fn roll_hash(old: i64, new: u8) -> i64 {
    (((old * PRIME) % MODULUS) + (new as i64)) % MODULUS
}

#[allow(dead_code)]
fn roll_hash_back(old: i64, new: u8) -> i64 {
    ((((old + MODULUS) - (new as i64)) % MODULUS) * PRIME_INVERSE) % MODULUS
}

#[inline(always)]
pub fn roll_hash_slice(slice: &[u8]) -> i64 {
    let mut hash = 0;
    for &byte in slice {
        hash = roll_hash(hash, byte);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_hash() {
        let result = roll_hash_back(roll_hash(roll_hash(0, 10), 17), 17);
        let r2 = roll_hash(0, 10);
        assert_eq!(result, r2);
    }
}