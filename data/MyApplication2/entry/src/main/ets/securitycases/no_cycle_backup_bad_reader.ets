import { BackupWriter } from './no_cycle_backup_bad_writer';

export class BackupReader {
  private writer: BackupWriter = new BackupWriter();

  readChunk(chunk: string): void {
    buffer.push(chunk);
    this.writer.flushChunk(this);
  }

  checksum(): number {
    return buffer.join('').length;
  }
}

const buffer: Array<string> = [];
export const reader: BackupReader = new BackupReader();
reader.readChunk('initial');
