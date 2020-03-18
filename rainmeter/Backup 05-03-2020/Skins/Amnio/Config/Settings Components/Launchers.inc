
[LaunchersArrow]
Meter=Image
ImageName=#@#Images\SettingsArrow.png
X=36
Y=56
ImageTint=#MainColour#,150

[LaunchersSettingsTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Text=Launchers
FontSize=30

[LaunchersTitleLine]
Meter=String
X=42
Y=50
W=400
H=1
SolidColor=#MainColour#,150

[Launcher1TextTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=58
Text=Launcher 1 text:
StringAlign=Right
FontSize=12

[Launcher1TextBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 1"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher1TextText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#Launcher1Text#
FontSize=12
FontColor=#TertiaryColour#,255

[Launcher1PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=80
Text=Launcher 1 path:
StringAlign=Right
FontSize=12

[Launcher1PathBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 2"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher1PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=158
H=18
Text=#Launcher1Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2

[Launcher1IconHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=102
Text=Launcher 1 hidden icon?
StringAlign=Right
FontSize=12

[Launcher1IconHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=0,0,0,1
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher1IconHidden][!SetVariable TickboxVariableCurrentValue #Launcher1IconHidden#][!SetVariable TickboxVariableFirstValue 1][!SetVariable TickboxVariableSecondValue 0][!SetVariable TickboxConfig "Launchers\Launcher 1"][!SetVariable TickboxSection Launcher1IconHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher1IconPathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=124
Text=Launcher 1 image:
StringAlign=Right
FontSize=12

[Launcher1IconPathImage]
Meter=Image
ImageName=#Launcher1IconPath#
X=12r
Y=r
W=40
H=40
ImageTint=#Launcher1IconTint#
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=["#@#Addons\RainFile\RainFile.exe" "File" "Variables" "Launcher1IconPath" "#@#Variables.inc" "#Launcher1IconPath#" "DPNE"]
MouseActionCursor=1

[Launcher1IconTintTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=376
Y=124
Text=Tint image?
StringAlign=Center
FontSize=12

[Launcher1IconTintTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=-9r
Y=20r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher1IconTint][!SetVariable TickboxVariableCurrentValue #Launcher1IconTint#][!SetVariable TickboxVariableFirstValue "#MainColour#,150"][!SetVariable TickboxVariableSecondValue 255,255,255,255][!SetVariable TickboxConfig "Launchers\Launcher 1"][!SetVariable TickboxSection Launcher1IconTintTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher2TextTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=184
Text=Launcher 2 text:
StringAlign=Right
FontSize=12

[Launcher2TextBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 3"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher2TextText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#Launcher2Text#
FontSize=12
FontColor=#TertiaryColour#,255

[Launcher2PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=206
Text=Launcher 2 path:
StringAlign=Right
FontSize=12

[Launcher2PathBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 4"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher2PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=158
H=18
Text=#Launcher2Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2

[Launcher2IconHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=228
Text=Launcher 2 hidden icon?
StringAlign=Right
FontSize=12

[Launcher2IconHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=0,0,0,1
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher2IconHidden][!SetVariable TickboxVariableCurrentValue #Launcher2IconHidden#][!SetVariable TickboxVariableFirstValue 1][!SetVariable TickboxVariableSecondValue 0][!SetVariable TickboxConfig "Launchers\Launcher 2"][!SetVariable TickboxSection Launcher2IconHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher2IconPathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=250
Text=Launcher 2 image:
StringAlign=Right
FontSize=12

[Launcher2IconPathImage]
Meter=Image
ImageName=#Launcher2IconPath#
X=12r
Y=r
W=40
H=40
ImageTint=#Launcher2IconTint#
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=["#@#Addons\RainFile\RainFile.exe" "File" "Variables" "Launcher2IconPath" "#@#Variables.inc" "#Launcher2IconPath#" "DPNE"]
MouseActionCursor=1

[Launcher2IconTintTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=376
Y=250
Text=Tint image?
StringAlign=Center
FontSize=12

[Launcher2IconTintTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=-9r
Y=20r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher2IconTint][!SetVariable TickboxVariableCurrentValue #Launcher2IconTint#][!SetVariable TickboxVariableFirstValue "#MainColour#,150"][!SetVariable TickboxVariableSecondValue 255,255,255,255][!SetVariable TickboxConfig "Launchers\Launcher 2"][!SetVariable TickboxSection Launcher2IconTintTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher3TextTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=310
Text=Launcher 3 text:
StringAlign=Right
FontSize=12

[Launcher3TextBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 5"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher3TextText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#Launcher3Text#
FontSize=12
FontColor=#TertiaryColour#,255

[Launcher3PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=332
Text=Launcher 3 path:
StringAlign=Right
FontSize=12

[Launcher3PathBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 6"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher3PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=158
H=18
Text=#Launcher3Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2

[Launcher3IconHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=354
Text=Launcher 3 hidden icon?
StringAlign=Right
FontSize=12

[Launcher3IconHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=0,0,0,1
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher3IconHidden][!SetVariable TickboxVariableCurrentValue #Launcher3IconHidden#][!SetVariable TickboxVariableFirstValue 1][!SetVariable TickboxVariableSecondValue 0][!SetVariable TickboxConfig "Launchers\Launcher 3"][!SetVariable TickboxSection Launcher3IconHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher3IconPathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=376
Text=Launcher 3 image:
StringAlign=Right
FontSize=12

[Launcher3IconPathImage]
Meter=Image
ImageName=#Launcher3IconPath#
X=12r
Y=r
W=40
H=40
ImageTint=#Launcher3IconTint#
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=["#@#Addons\RainFile\RainFile.exe" "File" "Variables" "Launcher3IconPath" "#@#Variables.inc" "#Launcher3IconPath#" "DPNE"]
MouseActionCursor=1

[Launcher3IconTintTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=376
Y=376
Text=Tint image?
StringAlign=Center
FontSize=12

[Launcher3IconTintTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=-9r
Y=20r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher3IconTint][!SetVariable TickboxVariableCurrentValue #Launcher3IconTint#][!SetVariable TickboxVariableFirstValue "#MainColour#,150"][!SetVariable TickboxVariableSecondValue 255,255,255,255][!SetVariable TickboxConfig "Launchers\Launcher 3"][!SetVariable TickboxSection Launcher3IconTintTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher4TextTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=436
Text=Launcher 4 text:
StringAlign=Right
FontSize=12

[Launcher4TextBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 7"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher4TextText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#Launcher4Text#
FontSize=12
FontColor=#TertiaryColour#,255

[Launcher4PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=458
Text=Launcher 4 path:
StringAlign=Right
FontSize=12

[Launcher4PathBackground]
Meter=String
X=12r
Y=r
W=160
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureLaunchersInputText "ExecuteBatch 8"]
MouseActionCursor=1
MouseActionCursorName=Text

[Launcher4PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=158
H=18
Text=#Launcher4Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2

[Launcher4IconHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=480
Text=Launcher 4 hidden icon?
StringAlign=Right
FontSize=12

[Launcher4IconHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=0,0,0,1
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher4IconHidden][!SetVariable TickboxVariableCurrentValue #Launcher4IconHidden#][!SetVariable TickboxVariableFirstValue 1][!SetVariable TickboxVariableSecondValue 0][!SetVariable TickboxConfig "Launchers\Launcher 4"][!SetVariable TickboxSection Launcher4IconHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[Launcher4IconPathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=502
Text=Launcher 4 image:
StringAlign=Right
FontSize=12

[Launcher4IconPathImage]
Meter=Image
ImageName=#Launcher4IconPath#
X=12r
Y=r
W=40
H=40
ImageTint=#Launcher4IconTint#
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=["#@#Addons\RainFile\RainFile.exe" "File" "Variables" "Launcher4IconPath" "#@#Variables.inc" "#Launcher4IconPath#" "DPNE"]
MouseActionCursor=1

[Launcher4IconTintTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=376
Y=502
Text=Tint image?
StringAlign=Center
FontSize=12

[Launcher4IconTintTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=-9r
Y=20r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable Launcher4IconTint][!SetVariable TickboxVariableCurrentValue #Launcher4IconTint#][!SetVariable TickboxVariableFirstValue "#MainColour#,150"][!SetVariable TickboxVariableSecondValue 255,255,255,255][!SetVariable TickboxConfig "Launchers\Launcher 4"][!SetVariable TickboxSection Launcher4IconTintTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
