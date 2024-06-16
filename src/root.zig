const std = @import("std");
const tst = std.testing;

/// Calculates the entropy H of the random variable described in rand_var
/// with each value of rand_var being the probability of a choice
/// Works on normalized probabilities
pub fn entropy(rand_var: []const f64) f64 {
    var h: f64 = 0.0;

    for (rand_var) |p| {
        h += p * @log2(p);
    }

    return -h;
}

/// Calculate the sum of the probabilities of the random variable
fn sumVar(rand_var: []const f64) f64 {
    var sum: f64 = 0.0;
    for (rand_var) |p| {
        sum += p;
    }
    return sum;
}

/// Normalizes the probabilities in rand_var so that their sum is 1
/// This is done in place
pub fn normalize(rand_var: []f64) void {
    // Get the sum of all the random variables
    const sum = sumVar(rand_var);
    // It shouldn't be 0
    std.debug.assert(sum != 0.0);
    // Normalize so the the sum will be 1.0
    for (rand_var) |*p| {
        p.* /= sum;
    }
}

/// Transforms in place a variable rand_var into a slice of ranges
/// where each range is an elem to the next (the first being 0 to rand_var[0])
/// for use in pick()
/// This is done in place so the result is replaced in rand_var
/// This cannot work on 0 elements
fn probabilitiesToRanges(rand_var: []f64) void {
    var sum: f64 = 0.0;
    for (rand_var) |*p| {
        sum += p.*;
        p.* = sum;
    }

    // Assert that we got to 1
    tst.expectApproxEqAbs(1.0, sum, 0.001) catch unreachable;

    // Fix the last element at 1 to be sure that pick always finds a solution
    rand_var[rand_var.len - 1] = 1.0;
}

/// Uses a discrete probability random variable rand_var to choose an index within it
/// rand_var needs to be normalized
/// The alloc is for internal use and nothing is owned by the caller
pub fn pick(rand: std.Random, alloc: std.mem.Allocator, rand_var: []const f64) !usize {
    const f = rand.float(f64);

    // Internal alloc for the ranges
    const ranges = try alloc.dupe(f64, rand_var);
    defer alloc.free(ranges);

    // Get ranges
    probabilitiesToRanges(ranges);

    // Find which range f falls within
    for (ranges, 0..) |upper, index| {
        if (f <= upper)
            return index;
    }

    // A range should have been found
    unreachable;
}

test "sum" {
    // Test that sumVar is sane
    try tst.expectApproxEqAbs(1.0, sumVar(&[_]f64{1.0}), 0.001);
    try tst.expectApproxEqAbs(6.0, sumVar(&[_]f64{ 1.0, 2.0, 3.0 }), 0.001);
    try tst.expectApproxEqAbs(0.0, sumVar(&[_]f64{}), 0.001);
    try tst.expectApproxEqAbs(0.0, sumVar(&[_]f64{ 1.0, 0.5, 0.25 }), 1.75);
}

test "normalize" {
    // Normalize some slices, check that the sum is 1
    var ps1 = [_]f64{ 1.0, 1.0, 2.0 };
    normalize(&ps1);
    try tst.expectApproxEqAbs(1.0, sumVar(&ps1), 0.001);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.25, 0.25, 0.5 }, &ps1, 0.001);

    var ps2 = [_]f64{10};
    normalize(&ps2);
    try tst.expectApproxEqAbs(1.0, sumVar(&ps2), 0.001);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{1}, &ps2, 0.001);
}

test "entropy calculation" {
    // Try the entropy calculation for a few situations
    try tst.expectApproxEqRel(0.0, entropy(&[_]f64{1.0}), 0.001);
    try tst.expectApproxEqRel(1.0, entropy(&([_]f64{0.5} ** 2)), 0.001);
    try tst.expectApproxEqRel(2.0, entropy(&([_]f64{0.25} ** 4)), 0.001);
    try tst.expectApproxEqRel(0.7219, entropy(&[_]f64{ 0.80, 0.20 }), 0.001);
}

test "probabilities to ranges" {
    // Try some distributions
    var ps1 = [_]f64{ 0.25, 0.25, 0.5 };
    probabilitiesToRanges(&ps1);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.25, 0.5, 1.0 }, &ps1, 0.001);

    var ps2 = [_]f64{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    probabilitiesToRanges(&ps2);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0 }, &ps2, 0.001);
}

test "pick" {
    const alloc = std.testing.allocator;
    var prng = std.rand.DefaultPrng.init(@bitCast(std.time.timestamp()));
    const rand = prng.random();

    const ps1 = [_]f64{ 0.5, 0.5 };
    var picks = [2]usize{ 0, 0 };

    for (0..1000) |_| {
        const choice = try pick(rand, alloc, &ps1);
        picks[choice] += 1;
    }

    try tst.expectEqual(1000, picks[0] + picks[1]);
    // We should get about 50/50 distribution
    try tst.expect(picks[0] >= 450 and picks[0] <= 550);

    // TODO: test more scenaris
}

fn expectApproxEqualSlicesAbs(comptime T: type, expected: []const T, actual: []const T, tolerance: T) !void {
    try tst.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try tst.expectApproxEqAbs(e, a, tolerance);
    }
}
