
[ColoursArrow]
Meter=Image
ImageName=#@#Images\SettingsArrow.png
X=36
Y=16
ImageTint=#MainColour#,150

[ColoursTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Text=Colo#Colour#rs
FontSize=30

[ColoursTitleLine]
Meter=String
X=42
Y=50
W=400
H=1
SolidColor=#MainColour#,150

[ColoursColourSchemeTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Y=58
Text=Select your colo#Colour#r scheme:
FontSize=14

[ColoursColourSchemeBlack]
Meter=Image
ImageName=#@#Images\Border.png
X=118
Y=88
W=100
H=40
ImageTint=#QuaternaryColour#,255
SolidColor=0,0,0,150
ScaleMargins=5,5,5,5
LeftMouseUpAction=[!WriteKeyValue Variables MainColour 0,0,0 "#@#Variables.inc"][!WriteKeyValue Variables SecondaryColour 0,0,0 "#@#Variables.inc"][!WriteKeyValue Variables TertiaryColour 255,255,255 "#@#Variables.inc"][!WriteKeyValue Variables QuaternaryColour 0,0,0 "#@#Variables.inc"][!WriteKeyValue Variables QuinaryColour 0,0,0 "#@#Variables.inc"][!Refresh *]
MouseActionCursor=1

[ColoursColourSchemeWhite]
Meter=Image
ImageName=#@#Images\Border.png
X=258
Y=88
W=100
H=40
ImageTint=#QuaternaryColour#,255
SolidColor=255,255,255,150
ScaleMargins=5,5,5,5
LeftMouseUpAction=[!WriteKeyValue Variables MainColour 255,255,255 "#@#Variables.inc"][!WriteKeyValue Variables SecondaryColour 255,255,255 "#@#Variables.inc"][!WriteKeyValue Variables TertiaryColour 0,0,0 "#@#Variables.inc"][!WriteKeyValue Variables QuaternaryColour 255,255,255 "#@#Variables.inc"][!WriteKeyValue Variables QuinaryColour 255,255,255 "#@#Variables.inc"][!Refresh *]
MouseActionCursor=1

[ColoursOrEditColoursManually]
Meter=String
MeterStyle=FontStyleVegurRegular
X=48
Y=148
Text=Or edit colo#Colour#rs manually
FontSize=12

[ColoursOrEditColoursManuallyLine]
Meter=String
X=214
Y=158
W=228
H=1
SolidColor=#MainColour#,150

[ColoursMainColourTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=236
Y=178
Text=Main colo#Colour#r:
StringAlign=Right
FontSize=12

[ColoursMainColourColour]
Meter=Image
ImageName=#@#Images\Border.png
X=12r
Y=-2r
W=50
H=26
ScaleMargins=5,5,5,5
ImageTint=#QuaternaryColour#
SolidColor=#MainColour#,150
LeftMouseUpAction=["#@#Addons\RainRGB4\RainRGB4.exe" "VarName=MainColour" "FileName=#@#Variables.inc"]
MouseActionCursor=1

[ColoursSecondaryColourTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=236
Y=208
Text=Second colo#Colour#r:
StringAlign=Right
FontSize=12

[ColoursSecondaryColourColour]
Meter=Image
ImageName=#@#Images\Border.png
X=12r
Y=-2r
W=50
H=26
ScaleMargins=5,5,5,5
ImageTint=#QuaternaryColour#,255
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=["#@#Addons\RainRGB4\RainRGB4.exe" "VarName=SecondaryColour" "FileName=#@#Variables.inc"]
MouseActionCursor=1

[ColoursTertiaryColourTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=236
Y=238
Text=Third colo#Colour#r:
StringAlign=Right
FontSize=12

[ColoursTertiaryColourColour]
Meter=Image
ImageName=#@#Images\Border.png
X=12r
Y=-2r
W=50
H=26
ScaleMargins=5,5,5,5
ImageTint=#QuaternaryColour#,255
SolidColor=#TertiaryColour#,255
LeftMouseUpAction=["#@#Addons\RainRGB4\RainRGB4.exe" "VarName=TertiaryColour" "FileName=#@#Variables.inc"]
MouseActionCursor=1

[ColoursQuaternaryColourTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=236
Y=268
Text=Fourth colo#Colour#r:
StringAlign=Right
FontSize=12

[ColoursQuaternaryColourColour]
Meter=Image
ImageName=#@#Images\Border.png
X=12r
Y=-2r
W=50
H=26
ScaleMargins=5,5,5,5
ImageTint=#QuaternaryColour#,255
SolidColor=#QuaternaryColour#,255
LeftMouseUpAction=["#@#Addons\RainRGB4\RainRGB4.exe" "VarName=QuaternaryColour" "FileName=#@#Variables.inc"]
MouseActionCursor=1

[ColoursQuinaryColourTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=236
Y=298
Text=Fifth colo#Colour#r:
StringAlign=Right
FontSize=12

[ColoursQuinaryColourColour]
Meter=Image
ImageName=#@#Images\Border.png
X=12r
Y=-2r
W=50
H=26
ScaleMargins=5,5,5,5
ImageTint=#QuaternaryColour#,255
SolidColor=#QuinaryColour#,60
LeftMouseUpAction=["#@#Addons\RainRGB4\RainRGB4.exe" "VarName=QuinaryColour" "FileName=#@#Variables.inc"]
MouseActionCursor=1
