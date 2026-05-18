import fs from 'node:fs';
import path from 'node:path';

export class Store {
  private data: Record<string, unknown> = {};

  constructor(private file: string) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (fs.existsSync(file)) {
      try {
        this.data = JSON.parse(fs.readFileSync(file, 'utf8'));
      } catch {
        this.data = {};
      }
    }
  }

  all(): Record<string, unknown> {
    return this.data;
  }

  get<T = unknown>(key: string): T | undefined {
    return this.data[key] as T | undefined;
  }

  set(key: string, value: unknown) {
    this.data[key] = value;
    this.flush();
  }

  merge(patch: Record<string, unknown>) {
    Object.assign(this.data, patch);
    this.flush();
  }

  private flush() {
    fs.writeFileSync(this.file, JSON.stringify(this.data, null, 2));
  }
}
