import { BackupReader } from './no_cycle_backup_bad_reader';

export class BackupWriter {
  flushChunk(reader?: BackupReader): void {
    if (reader) {
      reader.checksum();
    }
  }
}
