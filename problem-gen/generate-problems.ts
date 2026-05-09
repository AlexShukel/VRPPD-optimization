/**
 * @file generate-problems.ts
 * @description
 * Emits VRPPD problem instances under `./problems/<vCount>_<oCount>/<i>_<ts>.json`,
 * sampled from the master orders/vehicles datasets produced by
 * `generate-data.ts`.
 *
 * Two banks of size classes are supported:
 * - **small**: the 1×1 .. 7×7 grid (10 samples each → 490 instances). This is
 *   the original benchmark suite, capped at the largest size brute-force can
 *   solve in reasonable time.
 * - **large**: 10×10, 10×20, 20×50, 30×100, 50×200, 100×500 (20 samples each
 *   → 120 instances).
 *
 * Usage:
 *   pnpm generate:problems          # both banks (default)
 *   pnpm generate:problems:small    # only the small grid
 *   pnpm generate:problems:large    # only the large classes
 */

import fs from 'fs/promises';
import path from 'path';

import type { Order, Problem, Vehicle } from './src/types';

interface SizeClass {
    vCount: number;
    oCount: number;
    samples: number;
}

const SMALL_GRID: SizeClass[] = (() => {
    const out: SizeClass[] = [];
    for (let v = 1; v <= 7; ++v) {
        for (let o = 1; o <= 7; ++o) {
            out.push({ vCount: v, oCount: o, samples: 10 });
        }
    }
    return out;
})();

const LARGE_CLASSES: SizeClass[] = [
    { vCount: 10, oCount: 10, samples: 20 },
    { vCount: 10, oCount: 20, samples: 20 },
    { vCount: 20, oCount: 50, samples: 20 },
    { vCount: 30, oCount: 100, samples: 20 },
    { vCount: 50, oCount: 200, samples: 20 },
    { vCount: 100, oCount: 500, samples: 20 },
];

const ROOT = __dirname;
const dataDir = path.resolve(ROOT, 'data');
const problemsDir = path.resolve(ROOT, 'problems');

const getRandomSubset = <T>(items: T[], count: number): T[] => {
    if (count > items.length) {
        throw new Error(
            `cannot sample ${count} unique items from a pool of ${items.length}; regenerate seed data with more entries`,
        );
    }
    const selectedIndices = new Set<number>();
    const result: T[] = [];

    while (result.length < count) {
        const randomIndex = Math.floor(Math.random() * items.length);
        if (!selectedIndices.has(randomIndex)) {
            selectedIndices.add(randomIndex);
            result.push(items[randomIndex]);
        }
    }

    return result;
};

/**
 * Find the most recent `data/<prefix>_<timestamp>.json`. Replaces the previous
 * hard-coded path that drifted whenever `generate-data.ts` was re-run.
 */
async function findLatestDataset(prefix: string): Promise<string> {
    const entries = await fs.readdir(dataDir);
    const matches = entries
        .map(name => {
            const m = name.match(/^(.+)_(\d+)\.json$/);
            return m && m[1] === prefix ? { name, ts: Number(m[2]) } : null;
        })
        .filter((x): x is { name: string; ts: number } => x !== null)
        .sort((a, b) => b.ts - a.ts);

    if (matches.length === 0) {
        throw new Error(`No ${prefix}_<timestamp>.json found under ${dataDir}. Run \`pnpm generate:data\` first.`);
    }
    return path.resolve(dataDir, matches[0].name);
}

function parseMode(argv: string[]): {
    classes: SizeClass[];
    modeLabel: string;
} {
    const args = new Set(argv.slice(2));
    if (args.has('--small')) {
        return { classes: SMALL_GRID, modeLabel: 'small' };
    }
    if (args.has('--large')) {
        return { classes: LARGE_CLASSES, modeLabel: 'large' };
    }
    if (args.has('--all') || args.size === 0) {
        return { classes: [...SMALL_GRID, ...LARGE_CLASSES], modeLabel: 'all' };
    }
    throw new Error(`Unknown args: ${[...args].join(' ')}. Expected --small | --large | --all.`);
}

const main = async () => {
    const { classes, modeLabel } = parseMode(process.argv);
    const timestamp = Date.now();

    const [ordersJsonPath, vehiclesJsonPath] = await Promise.all([
        findLatestDataset('orders'),
        findLatestDataset('vehicles'),
    ]);

    const [ordersJsonRaw, vehiclesJsonRaw] = await Promise.all([
        fs.readFile(ordersJsonPath, 'utf-8'),
        fs.readFile(vehiclesJsonPath, 'utf-8'),
    ]);

    const allOrders: Order[] = JSON.parse(ordersJsonRaw);
    const allVehicles: Vehicle[] = JSON.parse(vehiclesJsonRaw);

    console.log(
        `mode=${modeLabel}  pool: ${allVehicles.length} vehicles, ${allOrders.length} orders` +
            `  source: ${path.relative(ROOT, vehiclesJsonPath)}`,
    );

    let totalEmitted = 0;
    for (const cls of classes) {
        if (cls.vCount > allVehicles.length) {
            throw new Error(
                `class ${cls.vCount}_${cls.oCount} requires ${cls.vCount} vehicles but pool has only ${allVehicles.length}`,
            );
        }
        if (cls.oCount > allOrders.length) {
            throw new Error(
                `class ${cls.vCount}_${cls.oCount} requires ${cls.oCount} orders but pool has only ${allOrders.length}`,
            );
        }

        const dirName = `${cls.vCount}_${cls.oCount}`;
        const targetDir = path.resolve(problemsDir, dirName);
        await fs.mkdir(targetDir, { recursive: true });

        const writes: Promise<void>[] = [];
        for (let i = 0; i < cls.samples; ++i) {
            const vehicles = getRandomSubset(allVehicles, cls.vCount);
            const orders = getRandomSubset(allOrders, cls.oCount);
            const problem: Problem = { vehicles, orders };
            const fileName = `${i}_${timestamp}.json`;
            writes.push(fs.writeFile(path.join(targetDir, fileName), JSON.stringify(problem, null, 2)));
        }

        await Promise.all(writes);
        totalEmitted += cls.samples;
        console.log(`  ${dirName}: ${cls.samples} instances`);
    }

    console.log(
        `\nDone. ${totalEmitted} instances across ${classes.length} size classes under ${path.relative(ROOT, problemsDir)}.`,
    );
};

main().catch(err => {
    console.error(err);
    process.exit(1);
});
