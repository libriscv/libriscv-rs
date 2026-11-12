//! Guards the toolchain assumption the dispatcher is built on: `become`
//! through function pointers is a guaranteed tail call, even unoptimized.
//! If a toolchain bump breaks this, these tests overflow the stack.

#[derive(Clone, Copy)]
struct Op {
    handler: Handler,
    imm: u64,
}

struct State {
    acc: u64,
    loops: u64,
}

type Handler = fn(&mut State, &[Op], usize, u64) -> u64;

const FAULT: u64 = u64::MAX;

fn add(s: &mut State, ops: &[Op], i: usize, count: u64) -> u64 {
    let Some(op) = ops.get(i) else { return FAULT };
    s.acc = s.acc.wrapping_add(op.imm);
    let Some(next) = ops.get(i + 1) else {
        return FAULT;
    };
    become (next.handler)(s, ops, i + 1, count + 1)
}

fn branch(s: &mut State, ops: &[Op], _i: usize, count: u64) -> u64 {
    if s.loops == 0 {
        return count + 1;
    }
    s.loops -= 1;
    let Some(first) = ops.first() else {
        return FAULT;
    };
    become (first.handler)(s, ops, 0, count + 1)
}

#[test]
fn indirect_become_does_not_grow_the_stack() {
    let mut ops = vec![
        Op {
            handler: add,
            imm: 3
        };
        8
    ];
    ops.push(Op {
        handler: branch,
        imm: 0,
    });
    // ~9M chained handler calls; without TCO this needs hundreds of MB of stack.
    let mut s = State {
        acc: 0,
        loops: 1_000_000,
    };
    let count = (ops[0].handler)(&mut s, &ops, 0, 0);
    assert_eq!(count, 9 * 1_000_001);
    assert_eq!(s.acc, 3 * 8 * 1_000_001);
}
