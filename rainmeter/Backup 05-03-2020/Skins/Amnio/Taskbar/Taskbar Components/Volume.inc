
[VolumeExpand]
Meter=Image
ImageName=#@#Images/Taskbar/TaskbarVolume_#VolumeState#.png
X=(#WorkAreaX# + ((#WorkAreaWidth# - 40) - [TimeExpand:W] - [PowerExpand:W]))
Y=(#WorkAreaY# + (#TaskbarOrientation# * 285))
ImageTint=#MainColour#,150
SolidColor=0,0,0,1
MouseOverAction=[!SetOption VolumeHover SolidColor #MainColour#,150][!UpdateMeter VolumeHover][!Redraw]
MouseLeaveAction=[!SetOption VolumeHover SolidColor 0,0,0,1][!UpdateMeter VolumeHover][!Redraw]
LeftMouseUpAction=[!HideMeterGroup DropDown][!ShowMeterGroup Volume][!Redraw]
RightMouseUpAction=[!CommandMeasure MeasureVolume ToggleMute][!Update]
DynamicVariables=1
Hidden=#TaskbarVolumeHidden#

[VolumeHover]
Meter=String
X=r
Y=(#TaskbarOrientation# = 0 ? 38 : 0)r
W=40
H=2
SolidColor=0,0,0,1

[VolumeContract]
Meter=String
MeasureName=MeasureDigitalVolume
X=r
Y=(0 - (#TaskbarOrientation# = 0 ? 38 : 0))r
W=40
H=40
FontColor=#MainColour#,150
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=[!HideMeterGroup Volume][!Redraw]
Group=DropDown | Volume
Hidden=1

[VolumeBackground]
Meter=String
X=-40r
Y=(#TaskbarOrientation# = 0 ? 40 : (0 - 285))r
W=80
H=285
SolidColor=#SecondaryColour#,15
Group=DropDown | Volume
Hidden=1

[VolumeBackground2]
Meter=String
X=r
Y=r
W=80
H=285
SolidColor=#SecondaryColour#,15
Group=DropDown | Volume
Hidden=1

[VolumeUp]
Meter=Image
ImageName=#@#Images\Volume_Up.png
X=34r
Y=10r
ImageTint=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureVolume "ChangeVolume +5"][!Update]
MouseActionCursor=1
DynamicVariables=1
Group=DropDown | Volume
Hidden=1

[VolumePercentage]
Meter=String
MeasureName=MeasureVolumePercent
MeterStyle=FontStyleVegurRegular
X=6r
Y=18r
StringAlign=CenterCenter
FontSize=14
DynamicVariables=1
Group=DropDown | Volume
Hidden=1

[VolumeDown]
Meter=Image
ImageName=#@#Images\Volume_Down.png
X=-6r
Y=10r
ImageTint=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureVolume "ChangeVolume -5"][!Update]
MouseActionCursor=1
DynamicVariables=1
Group=DropDown | Volume
Hidden=1

[VolumeBar]
Meter=Bar
MeasureName=MeasureVolume
X=4r
Y=12r
W=4
H=150
BarColor=#MainColour#,150
SolidColor=#SecondaryColour#,40
LeftMouseUpAction=[!SetVariable VolumePercentage "(100 - $MouseY:%$)"][!UpdateMeasure MeasureSetVolume]
MouseActionCursor=1
Group=DropDown | Volume
Hidden=1

[VolumeMuteIcon]
Meter=Image
ImageName=#@#Images/Taskbar/TaskbarVolume_#VolumeState#.png
X=-18r
Y=3R
ImageTint=#MainColour#,150
SolidColor=0,0,0,1
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=[!CommandMeasure MeasureVolume ToggleMute][!Update]
DynamicVariables=1
Group=DropDown | Volume
Hidden=1

[VolumeDivider]
Meter=String
X=-10r
Y=2R
W=60
H=1
SolidColor=#SecondaryColour#,15
Group=DropDown | Volume
Hidden=1

[VolumeMixer]
Meter=String
MeterStyle=FontStyleVegurRegular
MeasureName=MeasureDate
X=30r
Y=20r
W=80
H=36
Text=Mixer
StringAlign=CenterCenter
FontSize=12
SolidColor=0,0,0,1
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=["C:\Windows\System32\SndVol.exe"]
Group=DropDown | Volume
Hidden=1
