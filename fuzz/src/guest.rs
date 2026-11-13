//! Harness for the guest C++ containers (`guest/`): host code reading and
//! changing `std::string`, `std::vector`, `std::unordered_map` and
//! `std::variant` objects whose bytes (headers, pointers, sizes, nodes)
//! the guest controls.

use libriscv::guest::{
    CustomHash, GuestPtr, GuestStdString, GuestStdUnorderedMap, GuestStdVariant, GuestStdVector,
};
use libriscv::{Machine, MachineOptions};

/// Where the fuzz input is placed. Every container operation works on the
/// object at the start of it; its pointers usually point back into it.
/// The dictionary has this address and a few offsets from it.
pub const OBJECT: u64 = 0x10000;
/// At most this much input goes into guest memory.
const OBJECT_MAX: usize = 0x10000;
/// The native heap system calls (unused by the harness, but they set up
/// the arena).
const HEAP_SYSCALLS: usize = 200;

libriscv::guest_variant! {
    /// `std::variant<bool, int, double, std::string, std::vector<long>>`.
    #[derive(Debug, Clone, PartialEq)]
    enum Value {
        Bool(bool),
        Int(i32),
        Double(f64),
        Str(GuestStdString),
        Longs(GuestStdVector<i64>),
    }
}

type Str = GuestPtr<GuestStdString>;
type Longs = GuestPtr<GuestStdVector<i64>>;
type Strings = GuestPtr<GuestStdVector<GuestStdString>>;
type IntMap = GuestPtr<GuestStdUnorderedMap<u64, u64>>;
type StrMap = GuestPtr<GuestStdUnorderedMap<GuestStdString, GuestStdString>>;
type FastMap = GuestPtr<GuestStdUnorderedMap<u32, i64, CustomHash>>;
type Variant = GuestPtr<GuestStdVariant<Value>>;

/// A string longer than the small-string buffer, so it's allocated.
const LONG: &str = "a string too long for the small-string buffer";

/// One operation on the object at [`OBJECT`], viewed as one of the
/// container types: `op >> 3` picks the type, `op & 7` the operation.
/// Errors are expected; panics are bugs.
fn operate(m: &mut Machine, op: u8) -> libriscv::Result<()> {
    let arg = u64::from(op);
    match (op >> 3) % 7 {
        0 => {
            let s = Str::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = s.to_string(m)?;
                }
                1 => {
                    let _ = s.to_bytes(m)?;
                }
                2 => s.set_string(m, b"short")?,
                3 => s.set_string(m, LONG.as_bytes())?,
                4 => s.assign(m, LONG)?,
                5 => s.relocate(m)?,
                _ => s.free(m)?,
            }
        }
        1 => {
            let v = Longs::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.to_vec(m)?;
                }
                1 => v.push_back(m, &(arg as i64))?,
                2 => v.pop_back(m)?,
                3 => v.resize(m, (arg & 31) as usize)?,
                4 => v.reserve(m, (arg & 63) as usize)?,
                5 => {
                    let _ = v.at(m, (arg & 3) as usize)?.get(m)?;
                }
                6 => v.clear(m)?,
                _ => v.free(m)?,
            }
        }
        2 => {
            let v = Strings::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.to_string_vector(m)?;
                }
                1 => v.push_back(m, LONG)?,
                2 => v.push_back(m, "short")?,
                3 => v.pop_back(m)?,
                4 => v.resize(m, (arg & 7) as usize)?,
                5 => v.relocate(m)?,
                6 => v.clear(m)?,
                _ => v.free(m)?,
            }
        }
        3 => {
            let map = IntMap::new(OBJECT);
            let key = arg & 15;
            match op & 7 {
                0 => {
                    let _ = map.entries(m)?;
                }
                1 => {
                    let _ = map.to_map(m)?;
                }
                2 => {
                    let _ = map.find(m, &key)?;
                }
                3 => {
                    let _ = map.insert_or_assign(m, &key, &arg)?;
                }
                4 => {
                    let _ = map.erase(m, &key)?;
                }
                5 => map.set_max_load_factor(m, 0.25)?,
                6 => map.clear(m)?,
                _ => map.free(m)?,
            }
        }
        4 => {
            let map = StrMap::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = map.to_map(m)?;
                }
                1 => {
                    let _ = map.find(m, "key")?;
                }
                2 => {
                    let _ = map.insert_or_assign(m, "key", LONG)?;
                }
                3 => {
                    let _ = map.insert_or_assign(m, LONG, "value")?;
                }
                4 => {
                    let _ = map.erase(m, "key")?;
                }
                5 => {
                    let _ = map.count(m, LONG)?;
                }
                6 => map.relocate(m)?,
                _ => map.free(m)?,
            }
        }
        5 => {
            let map = FastMap::new(OBJECT);
            let key = (arg & 15) as u32;
            match op & 7 {
                0 => {
                    let _ = map.entries(m)?;
                }
                1 => {
                    let _ = map.at(m, &key)?.get(m)?;
                }
                2 => {
                    let _ = map.try_emplace(m, &key, &-1i64)?;
                }
                3 => {
                    let _ = map.insert_or_assign(m, &key, &(arg as i64))?;
                }
                4 => {
                    let _ = map.erase(m, &key)?;
                }
                5 => {
                    let _ = map.contains(m, &key)?;
                }
                6 => map.clear(m)?,
                _ => map.free(m)?,
            }
        }
        _ => {
            let v = Variant::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.get(m)?;
                }
                1 => {
                    let _ = v.index(m)?;
                }
                2 => {
                    let _ = v.alternative::<GuestStdString>(m)?.to_string(m)?;
                }
                3 => {
                    let _ = v.alternative::<GuestStdVector<i64>>(m)?.to_vec(m)?;
                }
                4 => {
                    let _ = v.emplace::<GuestStdString, _>(m, LONG)?;
                }
                5 => {
                    let _ = v.emplace::<i32, _>(m, &(arg as i32))?;
                }
                6 => v.relocate(m)?,
                _ => v.free(m)?,
            }
        }
    }
    Ok(())
}

/// A machine with a native heap and `memory` at [`OBJECT`].
fn containers_machine(memory: &[u8]) -> Option<Machine> {
    let memory = memory.get(..OBJECT_MAX).unwrap_or(memory);
    let options = MachineOptions {
        memory_max: 16 << 20,
        ..crate::options()
    };
    let mut m = Machine::new(&[], options).ok()?;
    crate::quiet(&mut m);
    m.setup_native_heap(HEAP_SYSCALLS, crate::HEAP, crate::HEAP_SIZE)
        .ok()?;
    m.copy_to_guest(OBJECT, memory).ok()?;
    Some(m)
}

/// `data[..4]` are four operations (see [`operate`]) on the object at
/// [`OBJECT`]; `data[4..]` is the guest memory there.
pub fn fuzz_guest_containers(data: &[u8]) {
    let Some((ops, memory)) = data.split_first_chunk::<4>() else {
        return;
    };
    let Some(mut m) = containers_machine(memory) else {
        return;
    };
    for &op in ops {
        let _ = operate(&mut m, op);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containers_are_built_in_guest_memory() {
        let mut m = containers_machine(&[0; 64]).unwrap();
        // Type 0 (std::string), operation 3: assign a long string.
        operate(&mut m, 3).unwrap();
        assert_eq!(Str::new(OBJECT).to_string(&m).unwrap(), LONG);
        // Type 1 (std::vector<long>) on fresh memory: push_back, to_vec.
        let mut m = containers_machine(&[0; 64]).unwrap();
        operate(&mut m, 8 | 1).unwrap();
        assert_eq!(Longs::new(OBJECT).to_vec(&m).unwrap(), [9]);
        // Type 3 (unordered_map<u64, u64>): insert_or_assign, find.
        let mut m = containers_machine(&[0; 64]).unwrap();
        operate(&mut m, 24 | 3).unwrap();
        assert_eq!(IntMap::new(OBJECT).size(&m).unwrap(), 1);
    }

    /// The first `guestfuzzer` finding: a cyclic node list in a map that
    /// claims billions of elements ran the host out of memory.
    #[test]
    fn cyclic_map_input_finishes() {
        const INPUT: [u8; 38] = [
            0x9b, 0x63, 0x2a, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8,
            0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8, 0xc8,
            0xc8, 0xc8, 0xc8, 0xc8, 0x30, 0x01, 0x10, 0x00, 0x18, 0x00,
        ];
        fuzz_guest_containers(&INPUT);
    }
}
