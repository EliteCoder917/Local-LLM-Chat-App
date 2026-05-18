/**
 * Minimal line-based diff using LCS (Longest Common Subsequence).
 * Good enough for previewing AI-suggested file edits — handles thousands of
 * lines in milliseconds and avoids a 200KB dep on the `diff` package.
 */

export type DiffOp = { kind: 'eq' | 'add' | 'del'; line: string };

export function lineDiff(a: string, b: string): DiffOp[] {
  const A = a.split('\n');
  const B = b.split('\n');
  const m = A.length;
  const n = B.length;

  // LCS table
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (A[i] === B[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  // Backtrace
  const out: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (A[i] === B[j]) {
      out.push({ kind: 'eq', line: A[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ kind: 'del', line: A[i] });
      i++;
    } else {
      out.push({ kind: 'add', line: B[j] });
      j++;
    }
  }
  while (i < m) { out.push({ kind: 'del', line: A[i++] }); }
  while (j < n) { out.push({ kind: 'add', line: B[j++] }); }
  return out;
}

export function diffStats(ops: DiffOp[]): { adds: number; dels: number } {
  let adds = 0;
  let dels = 0;
  for (const o of ops) {
    if (o.kind === 'add') adds++;
    else if (o.kind === 'del') dels++;
  }
  return { adds, dels };
}
