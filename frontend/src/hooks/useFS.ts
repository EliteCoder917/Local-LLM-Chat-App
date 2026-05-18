import { BACKEND_HTTP } from '../ipc/bridge';
import type { FileNode } from '../state/types';

export async function listDir(path: string): Promise<FileNode[]> {
  const r = await fetch(`${BACKEND_HTTP}/fs/list?path=${encodeURIComponent(path)}`);
  if (!r.ok) throw new Error(`listDir failed: ${r.status}`);
  return r.json();
}

export async function readFileText(path: string): Promise<string> {
  const r = await fetch(`${BACKEND_HTTP}/fs/read?path=${encodeURIComponent(path)}`);
  if (!r.ok) throw new Error(`readFile failed: ${r.status}`);
  return r.text();
}

export async function writeFileText(path: string, content: string): Promise<void> {
  const r = await fetch(`${BACKEND_HTTP}/fs/write`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, content }),
  });
  if (!r.ok) throw new Error(`writeFile failed: ${r.status}`);
}
