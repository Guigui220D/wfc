const std = @import("std");
const tst = std.testing;

// Returns the namespace with the right vector type
pub fn Wfc(comptime N: comptime_int, comptime ft: type) type {
    if (ft != f64 and ft != f32)
        @compileError("WFC weights have to be a floating point type");

    const IdT = std.math.IntFittingRange(0, N - 1);

    return struct {
        pub const VecT: type = @Vector(N, ft);
        pub const ArrT: type = [N]ft;

        /// Normalizes the probabilities in vec so that their sum is 1
        /// Negative probabilities are also snapped to 0
        /// This is done in place
        pub fn normalize(vec: VecT) VecT {
            // Eliminate negative values
            const pred = vec > @as(VecT, @splat(0.0));
            const vec2 = @select(ft, pred, vec, @as(VecT, @splat(0.0)));

            // Avoid division by 0
            const total = @reduce(.Add, vec2);
            if (total == 0.0)
                return vec2;

            // Normalize
            return vec2 / @as(VecT, @splat(total));
        }

        /// Transforms in place a variable vec into a slice of ranges
        /// where each range is an elem to the next (the first being 0 to rand_var[0])
        /// works on a normalized vector only, returns an error if there is 0 valid probability
        /// for use in pick(), the vector has to be normalized
        fn getRanges(norm_vec: VecT) !ArrT {
            // Accumulate (TODO: can this be done in simd?)
            var ret: ArrT = norm_vec;
            var acc: ft = 0.0;
            for (&ret) |*p| {
                acc += p.*;
                p.* = acc;
            }

            // Check that there are some possibilities
            if (acc == 0.0)
                return error.NoChoices;

            // Just an assertion to avoid forgetting the normalization step
            std.debug.assert(@abs(acc - 1.0) < 0.001);

            // Fix the last element at 1 to be sure that pick always finds a solution
            ret[N - 1] = 1.0;

            return ret;
        }

        /// Uses a discrete probability random variable vec to choose an index within it
        /// vec has to be normalized
        pub fn pick(rand: std.Random, norm_vec: VecT) !IdT {
            const f = rand.float(f64);

            // Get ranges
            const ranges = try getRanges(norm_vec);

            // Find which range f falls within
            for (ranges, 0..) |upper, index| {
                if (f <= upper)
                    return @intCast(index);
            }

            // A range should have been found
            unreachable;
        }

        /// Calculates the entropy H of the random variable described in vec
        /// with each value of rand_var being the probability of a choice
        /// Works on normalized probabilities
        pub fn entropy(norm_vec: VecT) ft {
            // Predicative for removing zeros from the log calculation
            const pred = norm_vec > @as(VecT, @splat(0.0));
            // Get each probability * its log2 as per the entropy formula (except the zeros)
            // (thats because although log2(0) -> -inf, n*log2(n) -> 0 when n->0
            const elems = norm_vec * @select(ft, pred, @log2(norm_vec), @as(VecT, @splat(0.0)));
            return -@reduce(.Add, elems);
        }
    };
}

const TestWfc = Wfc(4, f64);

test "normalize" {
    // Normalize some slices, check that the sum is 1
    const v1 = .{ 1.0, 1.0, 2.0, 0.0 };
    const n1 = TestWfc.normalize(v1);
    try tst.expectApproxEqAbs(1.0, @reduce(.Add, n1), 0.001);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.25, 0.25, 0.5, 0.0 }, &@as(TestWfc.ArrT, n1), 0.001);

    const v2 = .{ 10.0, 0.0, 0.0, 0.0 };
    const n2 = TestWfc.normalize(v2);
    try tst.expectApproxEqAbs(1.0, @reduce(.Add, n2), 0.001);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 1.0, 0.0, 0.0, 0.0 }, &@as(TestWfc.ArrT, n2), 0.001);

    const v3 = .{ 1.0, -1.0, 3.0, 0.0 };
    const n3 = TestWfc.normalize(v3);
    try tst.expectApproxEqAbs(1.0, @reduce(.Add, n3), 0.001);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.25, 0.0, 3.0 / 4.0, 0.0 }, &@as(TestWfc.ArrT, n3), 0.001);

    const v4 = .{ 0.0, -1.0, 0.0, -5.0 };
    const n4 = TestWfc.normalize(v4);
    try tst.expectApproxEqAbs(0.0, @reduce(.Add, n4), 0.001);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.0, 0.0, 0.0, 0.0 }, &@as(TestWfc.ArrT, n4), 0.001);
}

test "entropy calculation" {
    // Try the entropy calculation for a few situations
    try tst.expectApproxEqRel(0.0, TestWfc.entropy(.{ 0.0, 1.0, 0.0, 0.0 }), 0.001);
    try tst.expectApproxEqRel(1.0, TestWfc.entropy(.{ 0.0, 0.5, 0.5, 0.0 }), 0.001);
    try tst.expectApproxEqRel(2.0, TestWfc.entropy(@splat(0.25)), 0.001);
    try tst.expectApproxEqRel(0.7219, TestWfc.entropy(.{ 0.8, 0.2, 0.0, 0.0 }), 0.001);
}

test "probabilities to ranges" {
    // Try some distributions
    const ps1 = .{ 0.25, 0.25, 0.5, 0.0 };
    const ra1 = try TestWfc.getRanges(ps1);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.25, 0.5, 1.0, 1.0 }, &@as(TestWfc.ArrT, ra1), 0.001);

    const ps2 = [_]f64{ 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    const ra2 = try TestWfc.getRanges(ps2);
    try expectApproxEqualSlicesAbs(f64, &[_]f64{ 0.0, 1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0 }, &@as(TestWfc.ArrT, ra2), 0.001);
}

test "pick" {
    var prng = std.rand.DefaultPrng.init(@bitCast(std.time.timestamp()));
    const rand = prng.random();

    const ps1 = .{ 0.5, 0.5, 0.0, 0.0 };
    var picks = [2]usize{ 0, 0 };

    for (0..1000) |_| {
        const choice = try TestWfc.pick(rand, ps1);
        try tst.expect(choice < 2);
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
