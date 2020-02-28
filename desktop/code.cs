using System;
using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace Sample
{
    public enum WallpaperStyle : int
    {
        Tile, Center, Stretch, NoChange
    }

    public class DesktopHelper
    {
        //Declare functions in unmanaged APIs.
        private const int SPIF_UPDATEINIFILE = 0x0001;
        private const int SPIF_SENDWININICHANGE = 0x0002;
        private const int SPI_SETDESKWALLPAPER = 0x0014;        

        [DllImport("user32.dll", EntryPoint = "SystemParametersInfo", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool SystemParametersInfoSet(int uAction, int uParam, object lpvParam, int fuWinIni);
        [DllImport("user32.dll", EntryPoint = "SystemParametersInfo", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int SystemParametersInfoSetWallpaper(int uAction, int uParam, string lpvParam, int fuWinIni);

        //Helper functions

        private static void RemoveWallpaper() 
        {
            SystemParametersInfoSetWallpaper(SPI_SETDESKWALLPAPER, 0, "", SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE);
        }

        public static void SetWallpaper(string path, WallpaperStyle style)
        {
            if (path == string.Empty )
            {
                RemoveWallpaper();
            }
            else if (System.IO.File.Exists(path) && 
                (
                    path.Trim().ToLowerInvariant().EndsWith(".bmp")||
                    path.Trim().ToLowerInvariant().EndsWith(".jpg")||
                    path.Trim().ToLowerInvariant().EndsWith(".jpeg")
                ))
            {
                SystemParametersInfoSetWallpaper(SPI_SETDESKWALLPAPER, 0, path, SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE);
                RegistryKey key = Registry.CurrentUser.OpenSubKey("Control Panel\\Desktop", true);
                switch (style)
                {
                    case WallpaperStyle.Stretch:
                        key.SetValue(@"WallpaperStyle", "2");
                        key.SetValue(@"TileWallpaper", "0");
                        break;
                    case WallpaperStyle.Center:
                        key.SetValue(@"WallpaperStyle", "1");
                        key.SetValue(@"TileWallpaper", "0");
                        break;
                    case WallpaperStyle.Tile:
                        key.SetValue(@"WallpaperStyle", "1");
                        key.SetValue(@"TileWallpaper", "1");
                        break;
                    case WallpaperStyle.NoChange:
                        break;
                }
                key.Close();
            }
        }
    }
}