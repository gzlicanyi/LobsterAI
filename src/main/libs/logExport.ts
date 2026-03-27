import fs from 'fs';
import { pipeline } from 'stream/promises';
import yazl from 'yazl';

export type LogArchiveEntry = {
  archiveName: string;
  filePath: string;
};

export type ExportLogsZipInput = {
  outputPath: string;
  entries: LogArchiveEntry[];
};

export type ExportLogsZipResult = {
  missingEntries: string[];
};

const EXPORT_TIMEOUT_MS = 30_000;

export async function exportLogsZip(input: ExportLogsZipInput): Promise<ExportLogsZipResult> {
  const zipFile = new yazl.ZipFile();
  const missingEntries: string[] = [];

  for (const entry of input.entries) {
    try {
      if (fs.existsSync(entry.filePath) && fs.statSync(entry.filePath).isFile()) {
        zipFile.addFile(entry.filePath, entry.archiveName);
        continue;
      }
    } catch {
      // File became inaccessible between check and add — treat as missing
    }
    missingEntries.push(entry.archiveName);
    zipFile.addBuffer(Buffer.alloc(0), entry.archiveName);
  }

  const outputStream = fs.createWriteStream(input.outputPath);

  const pipelinePromise = pipeline(zipFile.outputStream, outputStream);
  zipFile.end();

  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error('Log export timed out')), EXPORT_TIMEOUT_MS);
  });

  try {
    await Promise.race([pipelinePromise, timeoutPromise]);
  } finally {
    clearTimeout(timer);
  }

  return { missingEntries };
}
