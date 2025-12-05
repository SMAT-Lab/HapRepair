import { ResilientBackupWriter } from './no_cycle_backup_safe_writer';

export class ResilientBackupReader {
  private writer: ResilientBackupWriter = new ResilientBackupWriter();

  readChunk(chunk: string): number {
    this.writer.flushChunk(chunk);
    return chunk.length;
  }
}

const safeReader: ResilientBackupReader = new ResilientBackupReader();
safeReader.readChunk('initial');
