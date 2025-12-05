export class ResilientBackupWriter {
  flushChunk(chunk: string): void {
    flushed.push(chunk);
  }
}

const flushed: Array<string> = [];
