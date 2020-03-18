
[MeasureTaskbarLauncherGroup]
Measure=String
String=#TaskbarLauncherGroup#
IfMatch=1-4
IfMatchAction=[!ShowMeterGroup "Launchers1-4"]
IfNotMatchAction=[!ShowMeterGroup "Launchers5-8"]
UpdateDivider=-1

[TaskbarArrow]
Meter=Image
ImageName=#@#Images\SettingsArrow.png
X=36
Y=136
ImageTint=#MainColour#,150

[TaskbarTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Text=Taskbar
FontSize=30

[TaskbarTitleLine]
Meter=String
X=42
Y=50
W=400
H=1
SolidColor=#MainColour#,150

[TaskbarOrientationTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Y=58
Text=Select your preferred position for the taskbar:
FontSize=14

[TaskbarTopBackground]
Meter=String
X=118
Y=90
W=100
H=40
SolidColor=0,0,0,1

[TaskbarTopTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=168
Y=110
W=100
H=40
Text=Top
StringAlign=CenterCenter
FontSize=18
MouseOverAction=[!SetOption TaskbarTopBackground SolidColor "#SecondaryColour#,15"][!UpdateMeter TaskbarTopBackground][!Redraw]
MouseLeaveAction=[!SetOption TaskbarTopBackground SolidColor 0,0,0,1][!UpdateMeter TaskbarTopBackground][!Redraw]
LeftMouseUpAction=[!WriteKeyValue Variables TaskbarOrientation 0 "#@#Variables.inc"][!Refresh "#RootConfig#\Taskbar"]
SolidColor=#SecondaryColour#,15
ClipString=2

[TaskbarBottomBackground]
Meter=String
X=258
Y=90
W=100
H=40
SolidColor=0,0,0,1

[TaskbarBottomTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=308
Y=110
W=100
H=40
Text=Bottom
StringAlign=CenterCenter
FontSize=18
MouseOverAction=[!SetOption TaskbarBottomBackground SolidColor "#SecondaryColour#,15"][!UpdateMeter TaskbarBottomBackground][!Redraw]
MouseLeaveAction=[!SetOption TaskbarBottomBackground SolidColor 0,0,0,1][!UpdateMeter TaskbarBottomBackground][!Redraw]
LeftMouseUpAction=[!WriteKeyValue Variables TaskbarOrientation 1 "#@#Variables.inc"][!Refresh "#RootConfig#\Taskbar"]
SolidColor=#SecondaryColour#,15
ClipString=2

[TaskbarLaunchers]
Meter=String
MeterStyle=FontStyleVegurRegular
X=48
Y=134
Text=Launchers
FontSize=12

[TaskbarLaunchersLine]
Meter=String
X=122
Y=144
W=320
H=1
SolidColor=#MainColour#,150

[TaskbarLaunchers1-4Background]
Meter=String
X=42
Y=152
W=200
H=18
SolidColor=#SecondaryColour#,15
Group=Launchers1-4
Hidden=1

[TaskbarLaunchers1-4]
Meter=String
MeterStyle=FontStyleVegurRegular
X=100r
Y=9r
W=200
H=18
Text=1 - 4
StringAlign=CenterCenter
FontSize=12
ClipString=2
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=[!WriteKeyValue Variables TaskbarLauncherGroup 1-4][!HideMeterGroup Launchers5-8][!ShowMeterGroup Launchers1-4][!Redraw]

[TaskbarLaunchers5-8Background]
Meter=String
X=242
Y=152
W=200
H=18
SolidColor=#SecondaryColour#,15
Group=Launchers5-8
Hidden=1

[TaskbarLaunchers5-8]
Meter=String
MeterStyle=FontStyleVegurRegular
X=100r
Y=9r
W=200
H=18
Text=5 - 8
StringAlign=CenterCenter
FontSize=12
ClipString=2
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=[!WriteKeyValue Variables TaskbarLauncherGroup 5-8][!HideMeterGroup Launchers1-4][!ShowMeterGroup Launchers5-8][!Redraw]

[TaskbarLauncher1NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=174
Text=Launcher 1 name:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher1NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 1"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher1NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=236
H=18
Text=#TaskbarLauncher1Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=1
Group=Launchers1-4
Hidden=1

[TaskbarLauncher1PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=194
Text=Launcher 1 path:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher1PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 2"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher1PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=236
H=18
Text=#TaskbarLauncher1Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher1Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher1Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher1Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher1Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=214
Text=Launcher 2 name:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 3"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher2Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=234
Text=Launcher 2 path:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 4"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher2Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher2Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher2Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher2Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher2Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=254
Text=Launcher 3 name:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 5"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher3Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=274
Text=Launcher 3 path:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 6"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher3Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher3Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher3Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher3Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher3Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=294
Text=Launcher 4 name:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 7"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher4Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=314
Text=Launcher 4 path:
StringAlign=Right
FontSize=12
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 8"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=236
H=18
Text=#TaskbarLauncher4Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers1-4
Hidden=1

[TaskbarLauncher4Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher4Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher4Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher4Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers1-4
Hidden=1

[TaskbarLauncher5NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=174
Text=Launcher 5 name:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher5NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 9"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher5NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher5Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher5PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=194
Text=Launcher 5 path:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher5PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 10"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher5PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher5Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher5Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher5Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher5Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher5Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=214
Text=Launcher 6 name:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 11"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher6Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=234
Text=Launcher 6 path:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 12"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher6Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher6Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher6Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher6Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher6Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=254
Text=Launcher 7 name:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 13"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher7Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=274
Text=Launcher 7 path:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 14"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher7Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher7Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher7Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher7Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher7Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8NameTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=294
Text=Launcher 8 name:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8NameBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 15"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8NameText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher8Name#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8PathTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=178
Y=314
Text=Launcher 8 path:
StringAlign=Right
FontSize=12
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8PathBackground]
Meter=String
X=8r
Y=r
W=236
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTaskbarInputText "ExecuteBatch 16"]
MouseActionCursor=1
MouseActionCursorName=Text
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8PathText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
W=234
H=18
Text=#TaskbarLauncher8Path#
FontSize=12
FontColor=#TertiaryColour#,255
ClipString=2
Group=Launchers5-8
Hidden=1

[TaskbarLauncher8Hidden]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=236r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarLauncher8Hidden][!SetVariable TickboxVariableCurrentValue #TaskbarLauncher8Hidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarLauncher8Hidden][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
Group=Launchers5-8
Hidden=1

[TaskbarNotificationArea]
Meter=String
MeterStyle=FontStyleVegurRegular
X=48
Y=336
Text=Notification area
FontSize=12

[TaskbarNotificationAreaLine]
Meter=String
X=162
Y=346
W=280
H=1
SolidColor=#MainColour#,150

[TaskbarBinHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=358
Text=Bin:
StringAlign=Right
FontSize=12

[TaskbarBinHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarBinHidden][!SetVariable TickboxVariableCurrentValue #TaskbarBinHidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarBinHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[TaskbarSystemHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=380
Text=System:
StringAlign=Right
FontSize=12

[TaskbarSystemHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarSystemHidden][!SetVariable TickboxVariableCurrentValue #TaskbarSystemHidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarSystemHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[TaskbarVolumeHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=402
Text=Volume:
StringAlign=Right
FontSize=12

[TaskbarVolumeHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarVolumeHidden][!SetVariable TickboxVariableCurrentValue #TaskbarVolumeHidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarVolumeHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[TaskbarPowerHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=424
Text=Power:
StringAlign=Right
FontSize=12

[TaskbarPowerHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarPowerHidden][!SetVariable TickboxVariableCurrentValue #TaskbarPowerHidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarPowerHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1

[TaskbarTimeHiddenTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=446
Text=Time:
StringAlign=Right
FontSize=12

[TaskbarTimeHiddenTickbox]
Meter=Image
ImageName=#@#Images\Tickbox.png
X=12r
Y=r
ImageTint=255,255,255,255
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable TickboxVariable TaskbarTimeHidden][!SetVariable TickboxVariableCurrentValue #TaskbarTimeHidden#][!SetVariable TickboxVariableFirstValue 0][!SetVariable TickboxVariableSecondValue 1][!SetVariable TickboxConfig "Taskbar"][!SetVariable TickboxSection TaskbarTimeHiddenTickbox][!EnableMeasure MeasureTickboxVariableChanger]
MouseActionCursor=1
