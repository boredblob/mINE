
;-------------------------------------------
; MEASURES
;-------------------------------------------
; TICKBOX

[MeasureTickboxVariableChanger]
Measure=String
String=#TickboxVariableCurrentValue#
IfMatch=#TickboxVariableFirstValue#
IfMatchAction=[!WriteKeyValue Variables #TickboxVariable# "#TickboxVariableSecondValue#" "#@#Variables.inc"][!SetVariable TickboxFormula 0][!EnableMeasure MeasureTickboxChanger]
IfNotMatchAction=[!WriteKeyValue Variables #TickboxVariable# "#TickboxVariableFirstValue#" "#@#Variables.inc"][!SetVariable TickboxFormula 1][!EnableMeasure MeasureTickboxChanger]
Disabled=1
DynamicVariables=1

[MeasureTickboxChanger]
Measure=Calc
Formula=#TickboxFormula#
IfCondition=[MeasureTickboxChanger] = 0
IfTrueAction=[!WriteKeyValue #TickboxSection# ImageTint "0,0,0,1" "#CurrentPath#Settings Components\#MenuPanel#.inc"][!Refresh][!Refresh "#RootConfig#\#TickboxConfig#"]
IfFalseAction=[!WriteKeyValue #TickboxSection# ImageTint "#TertiaryColour#,255" "#CurrentPath#Settings Components\#MenuPanel#.inc"][!Refresh][!Refresh "#RootConfig#\#TickboxConfig#"]
Disabled=1
DynamicVariables=1

;-------------------------------------------
; FONT TOGGLER

[MeasureFontToggle]
Measure=String
String=#CurrentFont#
IfMatch=Regular
IfMatchAction=[!WriteKeyValue Variables #FontToggleVariable# Light "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\#FontConfig#"]
IfNotMatchAction=[!WriteKeyValue Variables #FontToggleVariable# Regular "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\#FontConfig#"]
Disabled=1
DynamicVariables=1

;-------------------------------------------
; COLOURS

[MeasureColoursInputText]
Measure=Plugin
Plugin=InputText.dll
X=168
Y=178
W=150
H=20
FontSize=12
FontFace=Vegur Regular
FontColor=#TertiaryColour#
SolidColor=#QuaternaryColour#
AntiAlias=1
Command1=[!WriteKeyValue Variables MainColour "$UserInput$" "#@#Variables.inc"][!Refresh *] DefaultValue="#MainColour#"
Command2=[!WriteKeyValue Variables SecondaryColour "$UserInput$" "#@#Variables.inc"][!Refresh *] Y=218 DefaultValue="#SecondaryColour#"
Command3=[!WriteKeyValue Variables TertiaryColour "$UserInput$" "#@#Variables.inc"][!Refresh *] Y=258 DefaultValue="#TertiaryColour#"
Command4=[!WriteKeyValue Variables QuaternaryColour "$UserInput$" "#@#Variables.inc"][!Refresh *] Y=298 DefaultValue="#QuaternaryColour#"
Command5=[!WriteKeyValue Variables QuinaryColour "$UserInput$" "#@#Variables.inc"][!Refresh *] Y=338 DefaultValue="#QuinaryColour#"

;-------------------------------------------
; LAUNCHERS

[MeasureLaunchersInputText]
Measure=Plugin
Plugin=InputText.dll
X=248
Y=58
W=160
H=18
FontSize=12
FontFace=Vegur Regular
FontColor=#TertiaryColour#
SolidColor=#QuaternaryColour#
AntiAlias=1
Command1=[!WriteKeyValue Variables Launcher1Text "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 1"] DefaultValue="#Launcher1Text#"
Command2=[!WriteKeyValue Variables Launcher1Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 1"] Y=80 DefaultValue="#Launcher1Path#"
Command3=[!WriteKeyValue Variables Launcher2Text "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 2"] Y=184 DefaultValue="#Launcher2Text#"
Command4=[!WriteKeyValue Variables Launcher2Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 2"] Y=206 DefaultValue="#Launcher2Path#"
Command5=[!WriteKeyValue Variables Launcher3Text "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 3"] Y=310 DefaultValue="#Launcher3Text#"
Command6=[!WriteKeyValue Variables Launcher3Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 3"] Y=332 DefaultValue="#Launcher3Path#"
Command7=[!WriteKeyValue Variables Launcher4Text "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 4"] Y=436 DefaultValue="#Launcher4Text#"
Command8=[!WriteKeyValue Variables Launcher4Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Launchers\Launcher 4"] Y=458 DefaultValue="#Launcher4Path#"

;-------------------------------------------
; SYSTEM

[MeasureSystemInputText]
Measure=Plugin
Plugin=InputText.dll
X=248
Y=58
W=40
H=18
FontSize=12
FontFace=Vegur Regular
FontColor=#TertiaryColour#
SolidColor=#QuaternaryColour#
AntiAlias=1
Command1=[!WriteKeyValue Variables CPUSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\CPU"] DefaultValue="#CPUSize#"
Command2=[!WriteKeyValue Variables PowerSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\Power"] Y=112 DefaultValue="#PowerSize#"
Command3=[!WriteKeyValue Variables RAMSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\RAM"] Y=166 DefaultValue="#RAMSize#"
Command4=[!WriteKeyValue Variables HDD1Size "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\HDD 1"] Y=220 DefaultValue="#HDD1Size#"
Command5=[!WriteKeyValue Variables HDD1Drive "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\HDD 1"] Y=242 W=20 DefaultValue="#HDD1Drive#"
Command6=[!WriteKeyValue Variables HDD2Size "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\HDD 2"] Y=296 DefaultValue="#HDD2Size#"
Command7=[!WriteKeyValue Variables HDD2Drive "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\HDD 2"] Y=318 W=20 DefaultValue="#HDD2Drive#"
Command8=[!WriteKeyValue Variables HDD3Size "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\HDD 3"] Y=372 DefaultValue="#HDD3Size#"
Command9=[!WriteKeyValue Variables HDD3Drive "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\System\HDD 3"] Y=394 W=20 DefaultValue="#HDD3Drive#"

;-------------------------------------------
; TASKBAR

[MeasureTaskbarInputText]
Measure=Plugin
Plugin=InputText.dll
X=186
Y=174
W=236
H=18
FontSize=12
FontFace=Vegur Regular
FontColor=#TertiaryColour#
SolidColor=#QuaternaryColour#
AntiAlias=1
Command1=[!WriteKeyValue Variables TaskbarLauncher1Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] DefaultValue="#TaskbarLauncher1Name#"
Command2=[!WriteKeyValue Variables TaskbarLauncher1Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=194 DefaultValue="#TaskbarLauncher1Path#"
Command3=[!WriteKeyValue Variables TaskbarLauncher2Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=214 DefaultValue="#TaskbarLauncher2Name#"
Command4=[!WriteKeyValue Variables TaskbarLauncher2Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=234 DefaultValue="#TaskbarLauncher2Path#"
Command5=[!WriteKeyValue Variables TaskbarLauncher3Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=254 DefaultValue="#TaskbarLauncher3Name#"
Command6=[!WriteKeyValue Variables TaskbarLauncher3Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=274 DefaultValue="#TaskbarLauncher3Path#"
Command7=[!WriteKeyValue Variables TaskbarLauncher4Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=294 DefaultValue="#TaskbarLauncher4Name#"
Command8=[!WriteKeyValue Variables TaskbarLauncher4Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=314 DefaultValue="#TaskbarLauncher4Path#"
Command9=[!WriteKeyValue Variables TaskbarLauncher5Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] DefaultValue="#TaskbarLauncher5Name#"
Command10=[!WriteKeyValue Variables TaskbarLauncher5Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=194 DefaultValue="#TaskbarLauncher5Path#"
Command11=[!WriteKeyValue Variables TaskbarLauncher6Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=214 DefaultValue="#TaskbarLauncher6Name#"
Command12=[!WriteKeyValue Variables TaskbarLauncher6Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=234 DefaultValue="#TaskbarLauncher6Path#"
Command13=[!WriteKeyValue Variables TaskbarLauncher7Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=254 DefaultValue="#TaskbarLauncher7Name#"
Command14=[!WriteKeyValue Variables TaskbarLauncher7Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=274 DefaultValue="#TaskbarLauncher7Path#"
Command15=[!WriteKeyValue Variables TaskbarLauncher8Name "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=294 DefaultValue="#TaskbarLauncher8Name#"
Command16=[!WriteKeyValue Variables TaskbarLauncher8Path "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Taskbar"] Y=314 DefaultValue="#TaskbarLauncher8Path#"

;-------------------------------------------
; TIME

[MeasureTimeInputText]
Measure=Plugin
Plugin=InputText.dll
X=248
Y=58
W=40
H=18
FontSize=12
FontFace=Vegur Regular
FontColor=#TertiaryColour#
SolidColor=#QuaternaryColour#
AntiAlias=1
Command1=[!WriteKeyValue Variables AnalogueClockSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Analogue Clock"] DefaultValue="#AnalogueClockSize#"
Command2=[!WriteKeyValue Variables DateSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Date"] Y=112 DefaultValue="#DateSize#"
Command3=[!WriteKeyValue Variables DigitalClockSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Digital Clock"] Y=166 DefaultValue="#DigitalClockSize#"

[MeasureTimeAnalogueClockFaceShape]
Measure=String
String=#AnalogueClockFaceShape#
IfMatch=0
IfMatchAction=[!WriteKeyValue Variables AnalogueClockFaceShape 1 "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Analogue Clock"]
IfMatch2=1
IfMatchAction2=[!WriteKeyValue Variables AnalogueClockFaceShape 2 "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Analogue Clock"]
IfMatch3=2
IfMatchAction3=[!WriteKeyValue Variables AnalogueClockFaceShape 0 "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Analogue Clock"]
Disabled=1

[MeasureTimeDigitalClockLeadingZeroToggle]
Measure=String
String=#DigitalClockLeadingZero#
IfMatch=#
IfMatchAction=[!WriteKeyValue Variables DigitalClockLeadingZero "" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Digital Clock"]
IfNotMatchAction=[!WriteKeyValue Variables DigitalClockLeadingZero # "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Digital Clock"]
Disabled=1

[MeasureTimeDigitalClockFormatToggle]
Measure=String
String=#DigitalClockFormat#
IfMatch=I
IfMatchAction=[!WriteKeyValue Variables DigitalClockFormat "H" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Digital Clock"]
IfNotMatchAction=[!WriteKeyValue Variables DigitalClockFormat "I" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Time\Digital Clock"]
Disabled=1

;-------------------------------------------
; EXTRAS

[MeasureExtrasInputText]
Measure=Plugin
Plugin=InputText.dll
X=248
Y=58
W=40
H=18
FontSize=12
FontFace=Vegur Regular
FontColor=#TertiaryColour#
SolidColor=#QuaternaryColour#
AntiAlias=1
Command1=[!WriteKeyValue Variables GalleryWidth "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Gallery"] DefaultValue="#GalleryWidth#"
Command2=[!WriteKeyValue Variables GalleryHeight "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Gallery"] Y=80 DefaultValue="#GalleryHeight#"
Command3=[!WriteKeyValue Variables GallerySeconds "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Gallery"] Y=102 DefaultValue="#GallerySeconds#"
Command4=[!WriteKeyValue Variables MusicSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Music"] Y=156 DefaultValue="#MusicSize#"
Command5=[!WriteKeyValue Variables DownloadSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Network\Download"] Y=210 DefaultValue="#DownloadSize#"
Command6=[!WriteKeyValue Variables UploadSize "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Network\Upload"] Y=254 DefaultValue="#UploadSize#"
Command7=[!WriteKeyValue Variables NotesWidth "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Notes"] Y=308 DefaultValue="#NotesWidth#"
Command8=[!WriteKeyValue Variables NotesHeight "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Notes"] Y=330 DefaultValue="#NotesHeight#"
Command9=[!WriteKeyValue Variables NotesTitle "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Notes"] Y=352 W=194 DefaultValue="#NotesTitle#"
Command10=[!WriteKeyValue Variables VolumeWidth "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Volume"] Y=406 DefaultValue="#VolumeWidth#"
Command11=[!WriteKeyValue Variables WeatherCode "$UserInput$" "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Weather"][!Refresh "#RootConfig#\Weather\Tomorrow"][!Refresh "#RootConfig#\Weather\Day After"][!Refresh "#RootConfig#\Weather\Day After 2"] Y=460 W=80 DefaultValue="#WeatherCode#"

[MeasureExtrasUnitToggle]
Measure=String
String=#Unit#
IfMatch=m
IfMatchAction=[!WriteKeyValue Variables Unit f "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Weather"][!Refresh "#RootConfig#\Weather\Tomorrow"][!Refresh "#RootConfig#\Weather\Day After"][!Refresh "#RootConfig#\Weather\Day After 2"]
IfNotMatchAction=[!WriteKeyValue Variables Unit m "#@#Variables.inc"][!Refresh][!Refresh "#RootConfig#\Weather"][!Refresh "#RootConfig#\Weather\Tomorrow"][!Refresh "#RootConfig#\Weather\Day After"][!Refresh "#RootConfig#\Weather\Day After 2"]
Disabled=1
