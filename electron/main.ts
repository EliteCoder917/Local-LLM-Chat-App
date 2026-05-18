import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import path from 'node:path';
import { autoUpdater } from 'electron-updater';
import { PythonBridge } from './python-bridge';
import { PermissionManager } from './permissions';
import { Store } from './store';

const isDev = !app.isPackaged;
let win: BrowserWindow | null = null;
let bridge: PythonBridge;
let perms: PermissionManager;
let store: Store;

async function createWindow() {
  store = new Store(path.join(app.getPath('userData'), 'settings.json'));
  perms = new PermissionManager(store);

  bridge = new PythonBridge({
    backendDir: isDev
      ? path.join(__dirname, '..', 'backend')
      : path.join(process.resourcesPath, 'backend'),
    pythonDist: isDev ? null : path.join(process.resourcesPath, 'python-dist'),
    projectRoot: isDev ? path.join(__dirname, '..') : process.resourcesPath,
    port: 8765,
  });
  await bridge.start();

  win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#171717',
    title: 'Local AI Chat',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  // F12 toggles DevTools; Ctrl+Shift+I as a second binding for muscle memory.
  win.webContents.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown') return;
    const isToggle =
      input.key === 'F12' ||
      (input.control && input.shift && input.key.toLowerCase() === 'i');
    if (isToggle) {
      win?.webContents.toggleDevTools();
      event.preventDefault();
    }
  });

  if (isDev) {
    await win.loadURL('http://localhost:5173');
    // Temporary: open DevTools on launch during debugging.
    // Remove once we're confident the UI is healthy.
    win.webContents.openDevTools({ mode: 'detach' });
  } else {
    await win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  bridge.on('event', (evt) => win?.webContents.send('llm:event', evt));
  bridge.on('permission-request', async (req) => {
    const granted = await perms.requestInteractive(win!, req);
    bridge.sendPermissionResponse(req.id, granted);
  });

  // Auto-update wiring — only in packaged builds. In dev there's no
  // installer to replace and the GitHub release polling would just spam
  // 404s. The library reads its provider config from
  // electron-builder.yml's `publish:` block, which we point at
  // EliteCoder917/Local-LLM-Chat-App.
  if (!isDev) {
    autoUpdater.autoDownload = true;            // grab updates in the background
    autoUpdater.autoInstallOnAppQuit = true;    // apply on next clean shutdown

    autoUpdater.on('error', (err) => {
      console.warn('[auto-update] error:', err?.message ?? err);
    });
    autoUpdater.on('update-available', (info) => {
      // Notify the renderer so it can show a small toast — non-blocking.
      win?.webContents.send('app:update-available', {
        version: info.version,
        releaseNotes: info.releaseNotes ?? '',
      });
    });
    autoUpdater.on('update-downloaded', async (info) => {
      // The update is staged. Ask the user whether to install now (relaunches
      // the app) or wait until next quit (autoInstallOnAppQuit handles it).
      const { response } = await dialog.showMessageBox(win!, {
        type: 'info',
        title: 'Update ready',
        message: `Version ${info.version} is ready to install.`,
        detail: 'Restart now to apply the update, or keep working and it will install next time you quit the app.',
        buttons: ['Restart now', 'Later'],
        defaultId: 0,
        cancelId: 1,
      });
      if (response === 0) {
        autoUpdater.quitAndInstall();
      }
    });

    try {
      await autoUpdater.checkForUpdates();
    } catch (e) {
      console.warn('[auto-update] initial check failed:', e);
    }
  }
}

// ─── IPC ──────────────────────────────────────────────────────────────
ipcMain.handle('settings:get', () => store.all());
ipcMain.handle('settings:set', (_e, patch: Record<string, unknown>) => {
  store.merge(patch);
  bridge.notifySettings(store.all());
  return store.all();
});

ipcMain.handle('perms:get', () => perms.all());
ipcMain.handle('perms:set', (_e, key: string, value: boolean) => {
  perms.set(key, value);
  bridge.notifyPermissions(perms.all());
  return perms.all();
});

ipcMain.handle('fs:pickFolder', async () => {
  if (!win) return null;
  const r = await dialog.showOpenDialog(win, { properties: ['openDirectory'] });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('fs:pickFile', async (_e, filters?: { name: string; extensions: string[] }[], multi?: boolean) => {
  if (!win) return null;
  const r = await dialog.showOpenDialog(win, {
    properties: multi ? ['openFile', 'multiSelections'] : ['openFile'],
    filters: filters ?? [{ name: 'All Files', extensions: ['*'] }],
  });
  if (r.canceled) return null;
  return multi ? r.filePaths : r.filePaths[0];
});

ipcMain.handle('llm:send', async (_e, payload: unknown) => bridge.sendChat(payload));
ipcMain.handle('llm:cancel', async () => bridge.cancel());

app.whenReady().then(createWindow);

app.on('window-all-closed', async () => {
  try { await bridge?.stop(); } catch { /* ignore */ }
  if (process.platform !== 'darwin') app.quit();
});

app.on('web-contents-created', (_e, contents) => {
  contents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
});
