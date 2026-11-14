//! Port of `shared_rodata.hpp`/`shared_rodata.cpp` and of
//! `Memory::prepare_shared_rodata`/`publish_shared_rodata` (`memory.cpp`):
//! machines loaded from the same program map one sealed image of its
//! read-only memory over the low part of their arenas, so that its text and
//! rodata exist in host memory once.
//!
//! The image covers guest memory `[0, end)`, where `end` is the page-aligned
//! end of the read-only ELF segments (and at most the start of the first
//! writable one), zeroes below the program included. It is published by the
//! first machine once it has loaded the program (relocations and all), and
//! later machines with the same layout and contents attach it *before*
//! loading, and skip copying the segments it covers.
//!
//! The unsafe half (the arena mapping and the memfd) is
//! `libriscv_jit_rt::{GuestArena, RodataImage}`; the image is mapped
//! copy-on-write there, see its documentation. This module is safe. The
//! feature lives under `jit` because that crate does; without it the arena
//! is a `Vec<u8>` and nothing is shared.
//!
//! Differences from C++: sharing also works with virtual paging and N-bit
//! arenas (C++ turns it off there, as a store into its read-only mapping
//! would crash; here it makes a private copy of the page). The image is a
//! whole number of *host* pages (C++: guest pages).

#[cfg(feature = "jit")]
pub(crate) use enabled::*;

#[cfg(not(feature = "jit"))]
pub(crate) use disabled::*;

#[cfg(not(feature = "jit"))]
mod disabled {
    use crate::elf::ProgramHeader;
    use crate::error::Result;
    use crate::memory::Memory;
    use crate::options::MachineOptions;

    /// Guest memory: a plain zeroed vector.
    pub(crate) type Arena = Vec<u8>;

    pub(crate) fn new_arena(len: usize) -> std::io::Result<Arena> {
        Ok(vec![0u8; len])
    }

    pub(crate) const fn empty_arena() -> Arena {
        Vec::new()
    }

    /// Nothing is ever shared.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct SharedRodata;

    impl SharedRodata {
        pub(crate) const NONE: Self = Self;
    }

    impl Memory {
        pub(crate) const fn prepare_shared_rodata(
            &mut self,
            _options: &MachineOptions,
            _phdrs: &[ProgramHeader],
        ) -> Result<()> {
            Ok(())
        }
        pub(crate) const fn publish_shared_rodata(&mut self, _options: &MachineOptions) {}

        /// Port of `Memory::shared_rodata_end`: always 0 without the
        /// `jit` feature.
        pub const fn shared_rodata_end(&self) -> u64 {
            let SharedRodata = self.shared_rodata;
            0
        }
    }
}

#[cfg(feature = "jit")]
mod enabled {
    use std::collections::HashMap;
    use std::sync::{Arc, LazyLock, Mutex, PoisonError, Weak};

    use libriscv_jit_rt::{GuestArena, RodataImage};

    use crate::elf::{self, ProgramHeader};
    use crate::error::Result;
    use crate::memory::{Memory, PAGE_SIZE};
    use crate::options::MachineOptions;

    /// Guest memory: a mapping that an image can be attached to.
    pub(crate) type Arena = GuestArena;

    pub(crate) fn new_arena(len: usize) -> std::io::Result<Arena> {
        GuestArena::new(len)
    }

    pub(crate) const fn empty_arena() -> Arena {
        GuestArena::empty()
    }

    /// One read-only region of a program in guest memory (C++
    /// `RodataSegment`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct RodataSegment {
        vaddr: u64,
        size: u64,
    }

    /// The read-only layout of one program in one arena (C++ `RodataKey`).
    /// It only selects the candidates, whose contents are then compared
    /// byte for byte. Segments are sorted by address.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct RodataKey {
        rodata_end: u64,
        arena_size: u64,
        segments: Vec<RodataSegment>,
    }

    /// A machine's part in sharing: the layout it is a candidate for, and
    /// the image it has attached.
    #[derive(Debug)]
    pub(crate) struct SharedRodata {
        key: Option<RodataKey>,
        image: Option<Arc<RodataImage>>,
    }

    impl SharedRodata {
        pub(crate) const NONE: Self = Self {
            key: None,
            image: None,
        };
    }

    /// Every live image, by layout (C++ `rodata_images`). The cache only
    /// observes images: the last machine to drop one frees it. Two programs
    /// can share a layout, so one key can hold several images.
    static IMAGES: LazyLock<Mutex<HashMap<RodataKey, Vec<Weak<RodataImage>>>>> =
        LazyLock::new(Mutex::default);

    fn images() -> std::sync::MutexGuard<'static, HashMap<RodataKey, Vec<Weak<RodataImage>>>> {
        // The map stays consistent whatever panicked while holding it.
        IMAGES.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The first live image for `key` that `verify` accepts, dropping the
    /// dead ones (C++ `find_verified`). Caller holds the lock.
    fn find_verified(
        map: &mut HashMap<RodataKey, Vec<Weak<RodataImage>>>,
        key: &RodataKey,
        verify: impl Fn(&[u8]) -> bool,
    ) -> Option<Arc<RodataImage>> {
        let list = map.get_mut(key)?;
        list.retain(|w| w.strong_count() > 0);
        list.iter()
            .filter_map(Weak::upgrade)
            .find(|image| verify(image.bytes()))
    }

    /// Port of `create_shared_rodata_image`: an image of `src` for `key`,
    /// remembered for the next machine; an identical existing one if
    /// another machine created it meanwhile.
    fn create_image(key: &RodataKey, src: &[u8]) -> Option<Arc<RodataImage>> {
        let mut map = images();
        // Creating an image is rare and the lock is held: sweep the
        // programs that were loaded once and never again.
        map.retain(|_, list| {
            list.retain(|w| w.strong_count() > 0);
            !list.is_empty()
        });
        if let Some(existing) = find_verified(&mut map, key, |image| image == src) {
            return Some(existing);
        }
        let image = Arc::new(RodataImage::new(src).ok()?);
        map.entry(key.clone())
            .or_default()
            .push(Arc::downgrade(&image));
        Some(image)
    }

    /// One read-only segment, and where in the binary its bytes are.
    #[derive(Clone, Copy)]
    struct RoSeg {
        vaddr: u64,
        offset: u64,
        size: u64,
    }

    /// True when `bytes` is all zeroes.
    fn is_zeroed(bytes: Option<&[u8]>) -> bool {
        bytes.is_some_and(|b| b.iter().all(|&x| x == 0))
    }

    impl Memory {
        /// Port of `Memory::prepare_shared_rodata`: work out whether the
        /// program can share its read-only memory, and attach an existing
        /// image of it if there is one, which `binary_load_ph` then skips.
        /// Called before any segment is loaded. Never fails the load: a
        /// program that can't share just loads normally.
        pub(crate) fn prepare_shared_rodata(
            &mut self,
            options: &MachineOptions,
            phdrs: &[ProgramHeader],
        ) -> Result<()> {
            if !options.use_shared_rodata
                || !options.load_program
                || !self.uses_flat_memory_arena()
                || !libriscv_jit_rt::shared_rodata_supported()
            {
                return Ok(());
            }
            let bin_len = self.binary.len() as u64;
            let mut segments = Vec::new();
            let mut rodata_end = 0u64;
            let mut writable_begin = u64::MAX;
            for ph in phdrs.iter().filter(|ph| ph.p_type == elf::PT_LOAD) {
                let Ok(vaddr) = self.elf_base_address(ph.p_vaddr) else {
                    return Ok(());
                };
                // Always pick the lowest writable segment.
                if ph.p_flags & elf::PF_W != 0 {
                    writable_begin = writable_begin.min(vaddr);
                    continue;
                }
                if ph.p_filesz == 0 || ph.p_flags & elf::PF_R == 0 {
                    continue;
                }
                // binary_load_ph rejects segments outside the binary too,
                // but only runs after us.
                let Some(file_end) = ph.p_offset.checked_add(ph.p_filesz) else {
                    continue;
                };
                if file_end > bin_len {
                    continue;
                }
                let Some(end) = vaddr.checked_add(ph.p_filesz) else {
                    continue;
                };
                rodata_end = rodata_end.max(end);
                segments.push(RoSeg {
                    vaddr,
                    offset: ph.p_offset,
                    size: ph.p_filesz,
                });
            }
            if rodata_end == 0 || writable_begin < rodata_end {
                return Ok(());
            }

            // Only whole pages can be shared, and the host's pages may be
            // larger than the guest's.
            let page = (libriscv_jit_rt::host_page_size() as u64).max(PAGE_SIZE);
            let page_mask = !(page - 1);
            let mut shared_end = rodata_end & page_mask;
            if writable_begin != u64::MAX {
                shared_end = shared_end.min(writable_begin & page_mask);
            }
            if shared_end == 0 || shared_end > self.memory_arena_size() {
                return Ok(());
            }

            // Sorted by address, so that the layout doesn't depend on the
            // order of the program headers, and verify can walk it once.
            segments.sort_by_key(|s| s.vaddr);
            let shared: Vec<RoSeg> = segments
                .iter()
                .filter(|s| s.vaddr < shared_end)
                .map(|s| RoSeg {
                    size: s.size.min(shared_end - s.vaddr),
                    ..*s
                })
                .collect();
            // Nothing of the program itself lands inside the shared region.
            if shared.is_empty() {
                return Ok(());
            }
            let key = RodataKey {
                rodata_end: shared_end,
                arena_size: self.memory_arena_size(),
                segments: shared
                    .iter()
                    .map(|s| RodataSegment {
                        vaddr: s.vaddr,
                        size: s.size,
                    })
                    .collect(),
            };

            // An image belongs to this program only when every shared byte
            // matches: the program's segments, zeroes in between.
            let binary = Arc::clone(&self.binary);
            let verify = |image: &[u8]| -> bool {
                if image.len() as u64 != shared_end {
                    return false;
                }
                let mut pos = 0u64;
                for s in &shared {
                    if s.vaddr < pos {
                        return false;
                    }
                    let (Ok(p), Ok(v), Ok(o), Ok(n)) = (
                        usize::try_from(pos),
                        usize::try_from(s.vaddr),
                        usize::try_from(s.offset),
                        usize::try_from(s.size),
                    ) else {
                        return false;
                    };
                    let Some(img) = v.checked_add(n).and_then(|e| image.get(v..e)) else {
                        return false;
                    };
                    let Some(src) = o.checked_add(n).and_then(|e| binary.get(o..e)) else {
                        return false;
                    };
                    if !is_zeroed(image.get(p..v)) || img != src {
                        return false;
                    }
                    pos = s.vaddr + s.size;
                }
                usize::try_from(pos).is_ok_and(|p| is_zeroed(image.get(p..)))
            };
            let image = find_verified(&mut images(), &key, verify);
            self.shared_rodata.key = Some(key);
            // Attaching now lets the loader skip the segments it covers.
            if let Some(image) = image
                && self.arena.attach(&image).is_ok()
            {
                self.shared_rodata.image = Some(image);
                if options.verbose_loader {
                    println!("* Attached shared read-only memory of {shared_end} bytes");
                }
            }
            Ok(())
        }

        /// Port of `Memory::publish_shared_rodata`: offer the loaded
        /// read-only region to later machines, unless already sharing one.
        /// Called once the program is loaded and relocated.
        pub(crate) fn publish_shared_rodata(&mut self, options: &MachineOptions) {
            if self.shared_rodata.image.is_some() {
                return;
            }
            let Some(key) = &self.shared_rodata.key else {
                return;
            };
            let Some(src) = usize::try_from(key.rodata_end)
                .ok()
                .and_then(|n| self.arena.get(..n))
            else {
                return;
            };
            let Some(image) = create_image(key, src) else {
                return;
            };
            // Invisible to us: the bytes are the ones just copied out of here.
            if self.arena.attach(&image).is_ok() {
                self.shared_rodata.image = Some(image);
                if options.verbose_loader {
                    println!(
                        "* Created shared read-only memory of {} bytes",
                        self.shared_rodata_end()
                    );
                }
            }
        }

        /// Port of `Memory::shared_rodata_end`: the end of the shared
        /// read-only image this machine has attached, 0 if none.
        pub fn shared_rodata_end(&self) -> u64 {
            self.shared_rodata
                .image
                .as_ref()
                .map_or(0, |image| image.len() as u64)
        }
    }
}
