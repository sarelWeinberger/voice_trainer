/**
 * Google Apps Script — auto-link the VOICED audio files in the clinician
 * GRBAS sheet to their Google Drive URLs, so each row becomes click-to-play.
 *
 * ONE-TIME SETUP (see the Hebrew step-by-step your developer gave you):
 *   1. Upload the "voiced_clinician_audio" folder (208 .wav files) to Drive.
 *   2. Open the GRBAS Google Sheet  ->  Extensions  ->  Apps Script.
 *   3. Paste this whole file, replace AUDIO_FOLDER_ID below, and Save.
 *   4. Run "linkAudio" once (authorize when asked).
 *   5. Back in the sheet, a "▶ play" link appears in every audio_file cell.
 */

// === CONFIG =================================================================
// Paste the ID of the Drive folder that holds the 208 .wav files.
// Open the folder in Drive; the URL looks like
//   https://drive.google.com/drive/folders/THIS_PART_IS_THE_ID
const AUDIO_FOLDER_ID = 'PASTE_FOLDER_ID_HERE';

const SHEET_NAME = 'GRBAS Ratings';
const AUDIO_COL_HEADER = 'audio_file';
// ============================================================================


function linkAudio() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    throw new Error('Sheet "' + SHEET_NAME + '" not found.');
  }

  const values = sheet.getDataRange().getValues();
  const header = values[0];
  const col = header.indexOf(AUDIO_COL_HEADER); // 0-based
  if (col < 0) {
    throw new Error('Column "' + AUDIO_COL_HEADER + '" not found in row 1.');
  }

  // Build filename -> Drive URL map for everything in the audio folder.
  const folder = DriveApp.getFolderById(AUDIO_FOLDER_ID);
  const urlByName = {};
  const files = folder.getFiles();
  while (files.hasNext()) {
    const file = files.next();
    urlByName[file.getName()] =
      'https://drive.google.com/file/d/' + file.getId() + '/view';
  }

  let linked = 0;
  let missing = 0;
  for (let r = 1; r < values.length; r++) {
    const name = String(values[r][col] || '').trim();
    if (!name) { continue; }
    const url = urlByName[name];
    const cell = sheet.getRange(r + 1, col + 1);
    if (url) {
      cell.setFormula('=HYPERLINK("' + url + '", "▶ ' + name + '")');
      linked++;
    } else {
      missing++;
    }
  }

  SpreadsheetApp.getUi().alert(
    'Audio linking complete.\n' +
    'Linked: ' + linked + '\n' +
    'Missing (no matching file in the Drive folder): ' + missing);
}
