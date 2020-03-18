
[TimeExpand]
Meter=String
MeterStyle=FontStyleVegurRegular
MeasureName=MeasureDigitalTime
X=(#WorkAreaX# + (#WorkAreaWidth# - 35))
Y=((#WorkAreaY# + (#TaskbarOrientation# * 285)) + 20)
W=70
H=40
StringAlign=CenterCenter
FontSize=14
FontColor=#MainColour#,150
SolidColor=0,0,0,1
MouseOverAction=[!SetOption TimeHover SolidColor #MainColour#,150][!UpdateMeter TimeHover][!Redraw]
MouseLeaveAction=[!SetOption TimeHover SolidColor 0,0,0,1][!UpdateMeter TimeHover][!Redraw]
LeftMouseUpAction=[!HideMeterGroup DropDown][!ShowMeterGroup Time][!Redraw]
Hidden=#TaskbarTimeHidden#

[TimeHover]
Meter=String
X=-35r
Y=(#TaskbarOrientation# = 0 ? 18 : (0 - 20))r
W=70
H=2
SolidColor=0,0,0,1

[TimeContract]
Meter=String
X=r
Y=(0 - (#TaskbarOrientation# = 0 ? 38 : 0))r
W=70
H=40
SolidColor=#SecondaryColour#,15
LeftMouseUpAction=[!HideMeterGroup Time][!Redraw]
Group=DropDown | Time
Hidden=1

[TimeBackground]
Meter=String
X=-170r
Y=(#TaskbarOrientation# = 0 ? 40 : (0 - 170))r
W=240
H=170
SolidColor=#SecondaryColour#,15
Group=DropDown | Time
Hidden=1

[TimeBackground2]
Meter=String
X=r
Y=r
W=240
H=170
SolidColor=#SecondaryColour#,15
Group=DropDown | Time
Hidden=1

[TimeTime]
Meter=String
MeterStyle=FontStyleVegurLight
MeasureName=MeasureDigitalTime
X=120r
Y=46r
StringAlign=CenterCenter
FontSize=60
Group=DropDown | Time
Hidden=1

[TimeDate]
Meter=String
MeterStyle=FontStyleVegurRegular
MeasureName=MeasureDate
X=r
Y=46r
StringAlign=CenterCenter
FontSize=14
Group=DropDown | Time
Hidden=1

[TimeDivider]
Meter=String
X=-80r
Y=30r
W=160
H=1
SolidColor=#SecondaryColour#,15
Group=DropDown | Time
Hidden=1

[TimeSettings]
Meter=String
MeterStyle=FontStyleVegurRegular
X=80r
Y=24r
W=240
H=36
Text=Settings
StringAlign=CenterCenter
FontSize=12
SolidColor=0,0,0,1
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=[!ActivateConfig #RootConfig#\Config]
Group=DropDown | Time
Hidden=1
