import datetime
from win32_setctime import setctime
import os
# import piexif

folder = "./tickets/generated wip/"

from os import listdir
from os.path import isfile, join
files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

def updateImageTakenDate(file):
  year = 2022
  month = 4
  day = 13
  hour = 14
  minute = 23
  second = 12
  microsecond = 114

  date = datetime.datetime(year, month, day, hour, minute, second, microsecond)

  # exif_dict = piexif.load(file)
  # exif_date = date.strftime("%Y:%m:%d %H:%M:%S")
  # exif_dict['0th'][piexif.ImageIFD.DateTime] = exif_date
  # exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = exif_date
  # exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = exif_date
  # exif_bytes = piexif.dump(exif_dict)
  # piexif.insert(exif_bytes, file)

  timestamp = date.timestamp()
  os.utime(file, (timestamp, timestamp))

for f in files:
  print(f)
  updateImageTakenDate(f)