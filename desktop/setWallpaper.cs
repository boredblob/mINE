using System;
using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace WallpaperSetter
{
  class Program {

    private const int SPIF_UPDATEINIFILE = 0x0001;
    private const int SPIF_SENDWININICHANGE = 0x0002;
    private const int SPI_SETDESKWALLPAPER = 0x0014;        

    [DllImport("user32.dll", EntryPoint = "SystemParametersInfo", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern int SystemParametersInfoSetWallpaper(int uAction, int uParam, string lpvParam, int fuWinIni);

    public static void setWallpaper(string path) {
      SystemParametersInfoSetWallpaper(SPI_SETDESKWALLPAPER, 0, path, SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE);
    }

    public static void Main(string[] args) {
      if (args.Length < 1) {
        System.Console.WriteLine("Please enter a file path.");
        return;
      }
      
      setWallpaper(args[0]);
    }
  }
}