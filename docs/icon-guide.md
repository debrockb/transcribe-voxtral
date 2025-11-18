## Voxtral Launcher Icons

Fresh icon files now live under `assets/icons/`:

- `voxtral-icon-1024.png` (master artwork)
- `voxtral-icon-256.png`, `voxtral-icon-128.png`, … `voxtral-icon-16.png` for quick previews
- `voxtral-icon.icns` for macOS
- `voxtral-icon.ico` for Windows

### macOS – Assign to `Start Voxtral Web - Mac.command`
1. Open `assets/icons/voxtral-icon-1024.png` (or the `.icns`) in Preview.
2. Press `⌘A` followed by `⌘C` to copy the artwork.
3. In Finder, select `Start Voxtral Web - Mac.command`, press `⌘I` to open *Get Info*.
4. Click the small icon in the top-left of the info window so it highlights, then press `⌘V` to paste.

### Windows – Assign to `Start Voxtral Web - Windows.bat`
1. Right-click the `.bat` file and choose **Create shortcut**. Place it where you launch Voxtral.
2. Right-click the shortcut → **Properties** → **Shortcut** tab → **Change Icon…**
3. Browse to `assets/icons/voxtral-icon.ico`, select it, choose **OK**, then **Apply**.

Your launchers will now display the Voxtral node icon instead of the default script icons.
