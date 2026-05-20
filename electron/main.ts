import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import { createServer, AddressInfo } from 'node:net';
import fs from 'node:fs/promises';
import path from 'node:path';
import { autoUpdater } from 'electron-updater';
import { PythonBridge } from './python-bridge';
import { PermissionManager } from './permissions';
import { Store } from './store';

/** Return a TCP port we can bind to on 127.0.0.1, trying `preferred` first.
 *  Falls back to an OS-picked free port if `preferred` is held by a zombie
 *  uvicorn from a previous dev session (common after Ctrl+C). The brief
 *  unbind-before-spawn race is fine for a single-user local app — Python
 *  binds the same port a few ms later and nothing else is fighting for it. */
async function findFreePort(preferred: number): Promise<number> {
  const tryBind = (port: number) =>
    new Promise<number | null>((resolve) => {
      const srv = createServer();
      srv.unref();
      srv.once('error', () => resolve(null));
      srv.listen(port, '127.0.0.1', () => {
        const got = (srv.address() as AddressInfo).port;
        srv.close(() => resolve(got));
      });
    });
  const first = await tryBind(preferred);
  if (first != null) return first;
  const fallback = await tryBind(0);
  if (fallback == null) throw new Error('No free TCP port available on 127.0.0.1');
  console.warn(`[bridge] port ${preferred} in use, using ${fallback} instead`);
  return fallback;
}

// Holds the actual port we ended up spawning Python on, so the renderer can
// read it via sync IPC at preload time. Set inside createWindow().
let backendPort: number = 8765;

// Pin the userData folder to the legacy product name so existing installs
// (which wrote settings + conversations to %APPDATA%\Local AI IDE\) carry
// over after the rebrand to "Local AI Studio". Without this, Electron would
// derive the folder from the new productName and users would silently lose
// their saved chats. The display name in the title bar / installer / taskbar
// uses productName from electron-builder.yml; only the data folder is pinned.
app.setPath('userData', path.join(app.getPath('appData'), 'Local AI IDE'));

const isDev = !app.isPackaged;
let win: BrowserWindow | null = null;
let bridge: PythonBridge;
let perms: PermissionManager;
let store: Store;

async function createWindow() {
  store = new Store(path.join(app.getPath('userData'), 'settings.json'));
  perms = new PermissionManager(store);

  backendPort = await findFreePort(8765);

  bridge = new PythonBridge({
    backendDir: isDev
      ? path.join(__dirname, '..', 'backend')
      : path.join(process.resourcesPath, 'backend'),
    pythonDist: isDev ? null : path.join(process.resourcesPath, 'python-dist'),
    projectRoot: isDev ? path.join(__dirname, '..') : process.resourcesPath,
    port: backendPort,
    // Where Python's stdout+stderr get tee'd in packaged builds (where
    // they'd otherwise vanish into thin air). app.getPath('logs') is
    // the OS-conventional user-writable log location:
    //   • Windows: %APPDATA%\Local AI IDE\logs\
    //   • macOS:   ~/Library/Logs/Local AI IDE/
    //   • Linux:   ~/.config/Local AI IDE/logs/
    logDir: isDev ? undefined : app.getPath('logs'),
  });
  await bridge.start();

  win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#171717',
    title: 'Local AI Studio',
    autoHideMenuBar: true,
    // Hide the native OS title bar and let the React renderer draw its own
    // chrome — keeps the window blending with the app's dark theme instead
    // of the bright Windows / macOS frame. macOS still draws traffic lights
    // at the top-left because `titleBarStyle: 'hidden'` only hides the bar
    // itself, not the system buttons (which is what we want — they have
    // OS-level accessibility shortcuts).
    //
    // We deliberately do NOT set `titleBarOverlay`: that flag asks Windows
    // to overlay its OWN min/max/close buttons on the renderer, which would
    // fight with our custom React buttons (you'd see double controls plus
    // the system-accent border that Windows draws around them).
    frame: false,
    titleBarStyle: 'hidden',
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

  // Forward maximize state changes to the renderer so the custom title bar
  // can swap its maximize/restore icon without polling.
  win.on('maximize',   () => win?.webContents.send('window:maximized', true));
  win.on('unmaximize', () => win?.webContents.send('window:maximized', false));

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
// Sync handler used by preload to learn which port we picked. `ipcMain.on`
// + `event.returnValue` makes `ipcRenderer.sendSync(...)` work, which lets
// the renderer expose BACKEND_HTTP as a plain const instead of an async
// getter (`fetch(`${BACKEND_HTTP}/...`)` is called from ~25 places).
ipcMain.on('app:backendUrlSync', (e) => {
  e.returnValue = `http://127.0.0.1:${backendPort}`;
});

// Window control IPC — the custom React title bar drives these because we
// hid the native frame. Each one is a no-op if there's no window yet.
ipcMain.on('window:minimize',  () => win?.minimize());
ipcMain.on('window:maximize',  () => {
  if (!win) return;
  if (win.isMaximized()) win.unmaximize(); else win.maximize();
});
ipcMain.on('window:close',     () => win?.close());
// Renderer asks once at mount to sync its maximize-icon state with reality
// (e.g. after a snap-resize the OS performed without going through us).
ipcMain.on('window:isMaximizedSync', (e) => { e.returnValue = !!win?.isMaximized(); });

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

// Read a file from disk as base64 so the renderer (sandboxed, can't fetch
// file:// URLs) can turn it into a Blob/dataURI. Used by the attach-image
// menu — without this, picking an image via the OS dialog silently fails
// because fetch('file:///...') is blocked under Electron's CSP.
ipcMain.handle('fs:readFile', async (_e, filePath: string) => {
  const buf = await fs.readFile(filePath);
  return {
    base64: buf.toString('base64'),
    size: buf.byteLength,
  };
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
