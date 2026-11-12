//! Port of `page.hpp` and the shared pages of `memory_rw.cpp` (the zero
//! page, the guard page and the host code page).
//!
//! C++ pages hold a `unique_ptr<PageData>` that may be *non-owning*: arena
//! pages point into the flat arena, and the shared pages point at static
//! data. Here the page data is an enum instead, so no page ever aliases
//! another: [`PageData::Arena`] means "the bytes are in the arena at this
//! page's address", and the shared pages are `'static` arrays.

use super::PAGE_SIZE;

/// One page of guest memory.
pub(crate) type PageBytes = [u8; PAGE_SIZE as usize];

/// The shared zero page (C++ `Page::cow_page()`). The guard page shares its
/// backing, as in C++.
pub(crate) static ZERO_PAGE: PageBytes = [0; PAGE_SIZE as usize];

/// The host code page (C++ `Page::host_page()`): the exit function used as
/// the return address of vmcalls (STOP, then a jump back to it), followed by
/// the signal trampoline at `SIGRETURN_OFFSET`.
pub(crate) static HOST_PAGE: PageBytes = host_page();

// `i < CODE.len()`, which is less than the page size.
#[allow(clippy::indexing_slicing)]
const fn host_page() -> PageBytes {
    const CODE: [u8; 20] = [
        0x73, 0x00, 0xf0, 0x7f, // STOP: 0x7ff00073
        0x6f, 0xf0, 0xdf, 0xff, // JMP -4: 0xffdff06f
        // Signal trampoline, at Memory::SIGRETURN_OFFSET. Linux points the
        // return address of a signal handler at a trampoline like this one.
        0x93, 0x08, 0xb0, 0x08, // LI a7, 139 (rt_sigreturn)
        0x73, 0x00, 0x00, 0x00, // ECALL
        0x6f, 0xf0, 0xdf, 0xff, // JMP -4: 0xffdff06f
    ];
    let mut page = [0u8; PAGE_SIZE as usize];
    let mut i = 0;
    while i < CODE.len() {
        page[i] = CODE[i];
        i += 1;
    }
    page
}

/// Page protections and flags (C++ `PageAttributes`). The default is a
/// readable and writable page, as in C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageAttributes {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
    /// Writing makes a private copy first (the page is read-only until then).
    pub is_cow: bool,
    /// Skipped when forking (Phase 3).
    pub dont_fork: bool,
    /// Free for the embedder.
    pub user_defined: u8,
}

impl Default for PageAttributes {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            exec: false,
            is_cow: false,
            dont_fork: false,
            user_defined: 0,
        }
    }
}

impl PageAttributes {
    /// No access at all.
    pub const NONE: Self = Self {
        read: false,
        write: false,
        exec: false,
        is_cow: false,
        dont_fork: false,
        user_defined: 0,
    };

    /// From Linux `PROT_*` bits.
    pub const fn from_prot(prot: i32) -> Self {
        Self {
            read: prot & 1 != 0,
            write: prot & 2 != 0,
            exec: prot & 4 != 0,
            ..Self::NONE
        }
    }

    /// Read, write and execute flags.
    pub const fn rwx(read: bool, write: bool, exec: bool) -> Self {
        Self {
            read,
            write,
            exec,
            ..Self::NONE
        }
    }

    /// Linux `PROT_*` bits (C++ `to_prot`).
    pub const fn to_prot(self) -> i32 {
        self.read as i32 | (self.write as i32) << 1 | (self.exec as i32) << 2
    }

    /// C++ `is_default`: readable, writable and not executable.
    pub const fn is_default(self) -> bool {
        self.read && self.write && !self.exec
    }
}

/// Where a page's bytes live.
#[derive(Debug, Clone)]
pub(crate) enum PageData {
    /// In the flat arena, at the page's own address (C++: a non-owning page
    /// loaned from the arena).
    Arena,
    /// A page of its own.
    Owned(Box<PageBytes>),
    /// The shared zero page (C++ `cow_page`/`guard_page` data).
    Zero,
    /// The shared host code page.
    Host,
}

/// A page table entry (C++ `Page`).
#[derive(Debug, Clone)]
pub(crate) struct Page {
    pub attr: PageAttributes,
    pub data: PageData,
}

impl Page {
    /// C++ `Page::is_cow_page`: the page shows the shared zero page.
    pub(crate) const fn is_zero_page(&self) -> bool {
        matches!(self.data, PageData::Zero)
    }

    /// C++ `!attr.non_owning`: the page has its own data.
    pub(crate) const fn is_owned(&self) -> bool {
        matches!(self.data, PageData::Owned(_))
    }

    /// Human-readable protections (C++ `Page::to_string`).
    pub(crate) fn describe(attr: PageAttributes) -> String {
        let flag = |b: bool| if b { "[x]" } else { "[ ]" };
        format!(
            "Readable: {}  Writable: {}  Executable: {}",
            flag(attr.read),
            flag(attr.write),
            flag(attr.exec)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes() {
        assert!(PageAttributes::default().is_default());
        assert!(!PageAttributes::from_prot(7).is_default());
        assert_eq!(PageAttributes::from_prot(5).to_prot(), 5);
        assert_eq!(PageAttributes::from_prot(0), PageAttributes::NONE);
        assert_eq!(&HOST_PAGE[..4], &[0x73, 0x00, 0xf0, 0x7f]);
        assert_eq!(&HOST_PAGE[8..12], &[0x93, 0x08, 0xb0, 0x08]);
        assert!(HOST_PAGE[20..].iter().all(|&b| b == 0));
    }
}
