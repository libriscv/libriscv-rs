//! Laying out a region's control flow as structured Rust, after Norman
//! Ramsey's "Beyond Relooper: Recursive Translation of Unstructured Control
//! Flow to Structured Control Flow" (ICFP 2022), the algorithm WebAssembly
//! compilers use. C++'s translator writes `goto`s; Rust has none, so
//! forward branches become `break 'b<n>` out of labeled blocks, back edges
//! `continue 'l<n>` of loops, and LLVM sees plain branches. (A `loop { match
//! pc { .. } }` dispatcher is what [`dispatch`] writes instead: LLVM doesn't
//! thread the constant `pc` of direct transfers through the loop header, so
//! every transfer would be an indirect jump through the `match`, with all
//! the guest registers merged at one point.)
//!
//! # Entries
//!
//! Every label of a region can be entered from outside (the interpreter
//! begins blocks there, other regions jump there), which the layout models
//! as a root `match pc` with an edge to each label. Entering a loop
//! anywhere but at its header makes the control flow irreducible, which
//! structured code can't express. So such an entry goes to a copy of the
//! loop's body without its header, which runs the rest of the iteration and
//! then enters the loop at its header; loops nested in the copy are treated
//! the same way. Edges from the region's own code that enter a loop at the
//! side (statically irreducible control flow) use the copies too. Code only
//! grows with the loop nesting depth, and the copies are cold: they run at
//! most one partial iteration per entry.
//!
//! If the result still isn't reducible (it always should be; this is a
//! safety net), or grows too large, the region is written as a
//! [`dispatch`] loop instead, which is always correct.

use std::collections::HashMap;
use std::fmt::Write as _;

use super::emit::{Arm, Item};

/// Give up (and use [`dispatch`]) beyond this many nodes per label.
const MAX_GROWTH: usize = 8;

/// The region's control flow as a `loop { match pc { .. } }` dispatcher:
/// the root `match` adds the entry's block count, the inner one runs the
/// arms, whose transfers set `pc` and continue it.
pub(super) fn dispatch(arms: &[Arm], entry_counts: &[(u64, u64)]) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "'e: loop {{");
    let _ = writeln!(o, "match pc {{");
    for &(pc, k) in entry_counts {
        let _ = writeln!(o, "{pc:#x} => c = c.wrapping_add({k}),");
    }
    let _ = writeln!(o, "_ => break 'x A::I(pc),");
    let _ = writeln!(o, "}}");
    let _ = writeln!(o, "'r: loop {{");
    let _ = writeln!(o, "match pc {{");
    for arm in arms {
        let _ = writeln!(o, "{:#x} => {{", arm.pc);
        for item in &arm.items {
            match item {
                Item::Line(l) => {
                    let _ = writeln!(o, "{l}");
                }
                Item::Branch(t) => {
                    let _ = writeln!(o, "pc = {t:#x}; continue 'r;");
                }
            }
        }
        let _ = writeln!(o, "}}");
    }
    let _ = writeln!(
        o,
        "_ => break 'x A::X({}, pc),",
        crate::native_code::EXIT_JUMP
    );
    let _ = writeln!(o, "}}\n}}\n}}");
    o
}

/// A loop of a scope's control flow: its header and members (labels).
#[derive(Debug)]
struct Loop {
    header: usize,
    members: Vec<bool>,
}

/// A set of labels with their own nodes: the region itself, or the copy of
/// a loop's body without its header (inside the scope `parent`).
#[derive(Debug)]
struct Scope {
    parent: Option<usize>,
    /// For a copy: the loops of `parent` that enclose the copied loop. The
    /// copy is only entered from inside them, so it is inside them too:
    /// its exits into `parent` come from inside them.
    context: Vec<usize>,
    members: Vec<bool>,
    loops: Vec<Loop>,
    /// Per label, the loops containing it, outermost first.
    chains: Vec<Vec<usize>>,
    /// Label -> node, created on demand.
    nodes: HashMap<usize, usize>,
    /// Loop -> its copy's scope, created on demand.
    copies: HashMap<usize, usize>,
}

/// The labels' static control flow: successors (in item order) and
/// predecessors, by label index.
struct Flow {
    succ: Vec<Vec<usize>>,
    preds: Vec<Vec<usize>>,
    pcs: Vec<u64>,
}

impl Flow {
    /// The loops of the labels in `set` (recursively: the strongly
    /// connected components, then those of each without its header). A
    /// loop's header is the member entered from outside it, the one with
    /// the lowest address if there are several (the other entries then
    /// use copies) or none.
    fn loops(
        &self,
        set: &[bool],
        out: &mut Vec<Loop>,
        parents: &mut Vec<Option<usize>>,
        parent: Option<usize>,
    ) {
        for scc in self.sccs(set) {
            let single = scc.len() == 1;
            if single
                && !scc
                    .first()
                    .is_some_and(|&v| self.succ.get(v).is_some_and(|s| s.contains(&v)))
            {
                continue;
            }
            let mut members = vec![false; self.pcs.len()];
            for &v in &scc {
                if let Some(m) = members.get_mut(v) {
                    *m = true;
                }
            }
            let entered = |v: &usize| {
                self.preds
                    .get(*v)
                    .is_some_and(|p| p.iter().any(|&u| !members.get(u).copied().unwrap_or(false)))
            };
            let pc = |v: &usize| self.pcs.get(*v).copied().unwrap_or(u64::MAX);
            let header = scc
                .iter()
                .copied()
                .filter(entered)
                .min_by_key(pc)
                .or_else(|| scc.iter().copied().min_by_key(pc));
            let Some(header) = header else { continue };
            let id = out.len();
            let mut inner = members.clone();
            if let Some(h) = inner.get_mut(header) {
                *h = false;
            }
            out.push(Loop { header, members });
            parents.push(parent);
            self.loops(&inner, out, parents, Some(id));
        }
    }

    /// Tarjan's strongly connected components of the subgraph `set`.
    fn sccs(&self, set: &[bool]) -> Vec<Vec<usize>> {
        let n = self.pcs.len();
        let mut index = vec![usize::MAX; n];
        let mut low = vec![0; n];
        let mut on_stack = vec![false; n];
        let mut stack = Vec::new();
        let mut out = Vec::new();
        let mut next = 0;
        let inside = |v: usize| set.get(v).copied().unwrap_or(false);
        for root in (0..n).filter(|&v| inside(v)) {
            if index.get(root).is_none_or(|&i| i != usize::MAX) {
                continue;
            }
            // Iterative DFS: (node, next successor position).
            let mut work = vec![(root, 0usize)];
            while let Some(&mut (v, ref mut pos)) = work.last_mut() {
                if *pos == 0 {
                    if let (Some(i), Some(l)) = (index.get_mut(v), low.get_mut(v)) {
                        *i = next;
                        *l = next;
                    }
                    next += 1;
                    stack.push(v);
                    if let Some(s) = on_stack.get_mut(v) {
                        *s = true;
                    }
                }
                let succ = self.succ.get(v).map_or(&[][..], Vec::as_slice);
                if let Some(&w) = succ.get(*pos) {
                    *pos += 1;
                    if !inside(w) {
                        continue;
                    }
                    if index.get(w).is_some_and(|&i| i == usize::MAX) {
                        work.push((w, 0));
                    } else if on_stack.get(w).copied().unwrap_or(false) {
                        let iw = index.get(w).copied().unwrap_or(0);
                        if let Some(l) = low.get_mut(v) {
                            *l = (*l).min(iw);
                        }
                    }
                    continue;
                }
                work.pop();
                let lv = low.get(v).copied().unwrap_or(0);
                if let Some(&(u, _)) = work.last()
                    && let Some(l) = low.get_mut(u)
                {
                    *l = (*l).min(lv);
                }
                if index.get(v).is_some_and(|&i| i == lv) {
                    let mut scc = Vec::new();
                    while let Some(w) = stack.pop() {
                        if let Some(s) = on_stack.get_mut(w) {
                            *s = false;
                        }
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    out.push(scc);
                }
            }
        }
        out
    }

    fn scope(&self, parent: Option<usize>, context: Vec<usize>, members: Vec<bool>) -> Scope {
        let mut loops = Vec::new();
        let mut parents = Vec::new();
        self.loops(&members, &mut loops, &mut parents, None);
        let mut chains = vec![Vec::new(); self.pcs.len()];
        // Loops come in preorder (a loop before the loops inside it), so
        // pushing each loop onto its members' chains orders them outermost
        // first.
        for (id, l) in loops.iter().enumerate() {
            for (v, &m) in l.members.iter().enumerate() {
                if m && let Some(c) = chains.get_mut(v) {
                    c.push(id);
                }
            }
        }
        let _ = parents;
        Scope {
            parent,
            context,
            members,
            loops,
            chains,
            nodes: HashMap::new(),
            copies: HashMap::new(),
        }
    }
}

/// The graph being built: nodes are (scope, label) pairs; node 0 is the
/// root.
struct Builder<'f> {
    flow: &'f Flow,
    scopes: Vec<Scope>,
    /// Per node: its label (unused for the root).
    label: Vec<usize>,
    /// Per node: its scope.
    scope_of: Vec<usize>,
    /// Per node: the node each of its arm's branches goes to.
    succ: Vec<Vec<usize>>,
    work: Vec<usize>,
}

impl Builder<'_> {
    fn node(&mut self, scope: usize, label: usize) -> Option<usize> {
        if let Some(&n) = self.scopes.get(scope)?.nodes.get(&label) {
            return Some(n);
        }
        let n = self.label.len();
        self.label.push(label);
        self.scope_of.push(scope);
        self.succ.push(Vec::new());
        self.scopes.get_mut(scope)?.nodes.insert(label, n);
        self.work.push(n);
        Some(n)
    }

    /// The node a transfer to `target` goes to, from a node of `scope`
    /// inside the loops `inside` (of that scope; empty for the root).
    fn resolve(&mut self, mut scope: usize, inside: &[usize], target: usize) -> Option<usize> {
        let mut inside = inside.to_vec();
        loop {
            let s = self.scopes.get(scope)?;
            if !s.members.get(target).copied().unwrap_or(false) {
                // Leaving a copy: from inside the loops enclosing it.
                inside.clone_from(&s.context);
                scope = s.parent?;
                continue;
            }
            // The outermost loop entered from outside, not at its header.
            let side = s.chains.get(target)?.iter().copied().find(|&l| {
                !inside.contains(&l) && s.loops.get(l).is_some_and(|lp| lp.header != target)
            });
            let Some(l) = side else {
                return self.node(scope, target);
            };
            // Entering the copy, from outside its loops.
            scope = match s.copies.get(&l) {
                Some(&c) => c,
                None => {
                    let lp = s.loops.get(l)?;
                    let mut members = lp.members.clone();
                    *members.get_mut(lp.header)? = false;
                    // The loops enclosing `l`: those before it on its
                    // header's chain. (Every one of them is in `inside`,
                    // or `l` wouldn't be the outermost loop entered at the
                    // side.)
                    let chain = s.chains.get(lp.header)?;
                    let context = chain.iter().copied().take_while(|&e| e != l).collect();
                    let copy = self.flow.scope(Some(scope), context, members);
                    let c = self.scopes.len();
                    self.scopes.push(copy);
                    self.scopes.get_mut(scope)?.copies.insert(l, c);
                    c
                }
            };
            inside.clear();
        }
    }
}

/// The layout of a node graph: depth-first order and dominators.
struct Layout {
    succ: Vec<Vec<usize>>,
    rpo: Vec<usize>,
    /// Immediate dominator children, in reverse postorder.
    children: Vec<Vec<usize>>,
    merge: Vec<bool>,
    header: Vec<bool>,
}

impl Layout {
    fn new(succ: Vec<Vec<usize>>) -> Option<Self> {
        let n = succ.len();
        // Reverse postorder from the root (iterative DFS).
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut work = vec![(0usize, 0usize)];
        *seen.get_mut(0)? = true;
        while let Some(&mut (v, ref mut pos)) = work.last_mut() {
            if let Some(&w) = succ.get(v)?.get(*pos) {
                *pos += 1;
                if !seen.get(w).copied()? {
                    *seen.get_mut(w)? = true;
                    work.push((w, 0));
                }
            } else {
                post.push(v);
                work.pop();
            }
        }
        if post.len() != n {
            return None;
        }
        let order: Vec<usize> = post.iter().rev().copied().collect();
        let mut rpo = vec![0; n];
        for (i, &v) in order.iter().enumerate() {
            *rpo.get_mut(v)? = i;
        }
        let mut preds = vec![Vec::new(); n];
        for (v, ss) in succ.iter().enumerate() {
            for &w in ss {
                preds.get_mut(w)?.push(v);
            }
        }
        // Cooper, Harvey and Kennedy's iterative dominators.
        let mut idom = vec![usize::MAX; n];
        *idom.get_mut(0)? = 0;
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| -> Option<usize> {
            while a != b {
                while rpo.get(a)? > rpo.get(b)? {
                    a = *idom.get(a)?;
                }
                while rpo.get(b)? > rpo.get(a)? {
                    b = *idom.get(b)?;
                }
            }
            Some(a)
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &v in order.iter().skip(1) {
                let mut new = usize::MAX;
                for &p in preds.get(v)? {
                    if *idom.get(p)? == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        p
                    } else {
                        intersect(&idom, p, new)?
                    };
                }
                if *idom.get(v)? != new {
                    *idom.get_mut(v)? = new;
                    changed = true;
                }
            }
        }
        let dominates = |a: usize, mut b: usize| -> Option<bool> {
            loop {
                if a == b {
                    return Some(true);
                }
                if b == 0 {
                    return Some(false);
                }
                b = *idom.get(b)?;
            }
        };
        let mut merge = vec![false; n];
        let mut header = vec![false; n];
        for (v, ps) in preds.iter().enumerate() {
            let mut forward = 0;
            for &p in ps {
                if rpo.get(p)? >= rpo.get(v)? {
                    // A back edge must go to a dominator: reducible.
                    if !dominates(v, p)? {
                        return None;
                    }
                    *header.get_mut(v)? = true;
                } else {
                    forward += 1;
                }
            }
            *merge.get_mut(v)? = forward >= 2;
        }
        let mut children = vec![Vec::new(); n];
        for &v in order.iter().skip(1) {
            children.get_mut(*idom.get(v)?)?.push(v);
        }
        Some(Self {
            succ,
            rpo,
            children,
            merge,
            header,
        })
    }
}

/// Renders the structured code.
struct Writer<'a> {
    layout: &'a Layout,
    /// Per node: its arm (the root has none).
    arm: &'a [Option<&'a Arm>],
    /// The root's `match` arms, without their branch.
    root: &'a [String],
    out: String,
}

impl Writer<'_> {
    fn do_tree(&mut self, x: usize) -> Option<()> {
        let mut merges: Vec<usize> = self
            .layout
            .children
            .get(x)?
            .iter()
            .copied()
            .filter(|&c| self.layout.merge.get(c).copied().unwrap_or(false))
            .collect();
        // The highest reverse postorder first: the outermost block, whose
        // code comes last.
        merges.sort_unstable_by_key(|&c| std::cmp::Reverse(self.layout.rpo.get(c).copied()));
        if self.layout.header.get(x).copied()? {
            let _ = writeln!(self.out, "'l{x}: loop {{");
            self.node_within(x, &merges)?;
            let _ = writeln!(self.out, "}}");
        } else {
            self.node_within(x, &merges)?;
        }
        Some(())
    }

    fn node_within(&mut self, x: usize, merges: &[usize]) -> Option<()> {
        if let Some((&y, rest)) = merges.split_first() {
            let _ = writeln!(self.out, "'b{y}: {{");
            self.node_within(x, rest)?;
            let _ = writeln!(self.out, "}}");
            return self.do_tree(y);
        }
        let succ = self.layout.succ.get(x)?.clone();
        match self.arm.get(x)? {
            None => {
                // The root: the entry dispatch.
                let _ = writeln!(self.out, "match pc {{");
                for (line, &target) in self.root.iter().zip(&succ) {
                    let _ = writeln!(self.out, "{line}");
                    self.do_branch(x, target)?;
                    let _ = writeln!(self.out, "}}");
                }
                let _ = writeln!(self.out, "_ => break 'x A::I(pc),\n}}");
            }
            Some(arm) => {
                let mut targets = succ.iter();
                for item in &arm.items {
                    match item {
                        Item::Line(l) => {
                            let _ = writeln!(self.out, "{l}");
                        }
                        Item::Branch(_) => self.do_branch(x, *targets.next()?)?,
                    }
                }
            }
        }
        Some(())
    }

    fn do_branch(&mut self, from: usize, to: usize) -> Option<()> {
        let l = self.layout;
        if l.rpo.get(to)? <= l.rpo.get(from)? {
            let _ = writeln!(self.out, "continue 'l{to};");
        } else if l.merge.get(to).copied()? {
            let _ = writeln!(self.out, "break 'b{to};");
        } else {
            self.do_tree(to)?;
        }
        Some(())
    }
}

/// The region's control flow as structured code, or `None` if it can't be
/// laid out (then use [`dispatch`]). `entry_counts` are the labels with
/// the block count entering there adds.
pub(super) fn structure(arms: &[Arm], entry_counts: &[(u64, u64)]) -> Option<String> {
    let index: HashMap<u64, usize> = arms.iter().enumerate().map(|(i, a)| (a.pc, i)).collect();
    let n = arms.len();
    let mut succ = vec![Vec::new(); n];
    let mut preds = vec![Vec::new(); n];
    for (i, arm) in arms.iter().enumerate() {
        for item in &arm.items {
            if let Item::Branch(t) = item {
                let j = *index.get(t)?;
                succ.get_mut(i)?.push(j);
                preds.get_mut(j)?.push(i);
            }
        }
    }
    let flow = Flow {
        succ,
        preds,
        pcs: arms.iter().map(|a| a.pc).collect(),
    };
    let top = flow.scope(None, Vec::new(), vec![true; n]);
    let mut b = Builder {
        flow: &flow,
        scopes: vec![top],
        label: vec![usize::MAX],
        scope_of: vec![0],
        succ: vec![Vec::new()],
        work: Vec::new(),
    };
    // The root: every label is an entry.
    let mut root = Vec::with_capacity(entry_counts.len());
    let mut entries = Vec::with_capacity(entry_counts.len());
    for &(pc, k) in entry_counts {
        let label = *index.get(&pc)?;
        entries.push(b.resolve(0, &[], label)?);
        root.push(format!("{pc:#x} => {{ c = c.wrapping_add({k});"));
    }
    *b.succ.get_mut(0)? = entries;
    while let Some(v) = b.work.pop() {
        if b.label.len() > MAX_GROWTH * n + 16 {
            return None;
        }
        let (label, scope) = (*b.label.get(v)?, *b.scope_of.get(v)?);
        let inside = b.scopes.get(scope)?.chains.get(label)?.clone();
        let targets = flow.succ.get(label)?.clone();
        let mut out = Vec::with_capacity(targets.len());
        for t in targets {
            out.push(b.resolve(scope, &inside, t)?);
        }
        *b.succ.get_mut(v)? = out;
    }
    let arm: Vec<Option<&Arm>> = b.label.iter().map(|&l| arms.get(l)).collect();
    let layout = Layout::new(b.succ)?;
    let mut w = Writer {
        layout: &layout,
        arm: &arm,
        root: &root,
        out: String::new(),
    };
    // The root is a loop, which intra-region indirect jumps continue.
    w.out.push_str("'e: loop {\n");
    w.do_tree(0)?;
    w.out.push_str("}\n");
    Some(w.out)
}
