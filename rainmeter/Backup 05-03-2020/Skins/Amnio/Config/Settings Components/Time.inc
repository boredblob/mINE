
[MeasureTimeDigitalClockLeadingZero]
Measure=String
String=#DigitalClockLeadingZero#
IfMatch=#
IfMatchAction=[!ShowMeter TimeDigitalClockLeadingZeroTextYes]
IfNotMatchAction=[!ShowMeter TimeDigitalClockLeadingZeroTextNo]
UpdateDivider=-1

[MeasureTimeDigitalClockFormat]
Measure=String
String=#DigitalClockFormat#
IfMatch=I
IfMatchAction=[!ShowMeter TimeDigitalClockFormatText12h]
IfNotMatchAction=[!ShowMeter TimeDigitalClockFormatText24h]
UpdateDivider=-1

[TimeArrow]
Meter=Image
ImageName=#@#Images\SettingsArrow.png
X=36
Y=176
ImageTint=#MainColour#,150

[TimeTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Text=Time
FontSize=30

[TimeTitleLine]
Meter=String
X=42
Y=50
W=400
H=1
SolidColor=#MainColour#,150

[TimeAnalogueClockSizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=58
Text=Analog#Analogue# clock size:
StringAlign=Right
FontSize=12

[TimeAnalogueClockSizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTimeInputText "ExecuteBatch 1"]
MouseActionCursor=1
MouseActionCursorName=Text

[TimeAnalogueClockSizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#AnalogueClockSize#
FontSize=12
FontColor=#TertiaryColour#,255

[TimeAnalogueClockFaceShapeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=80
Text=Analog#Analogue# clock face shape:
StringAlign=Right
FontSize=12

[TimeAnalogueClockFaceShapeBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!EnableMeasure MeasureTimeAnalogueClockFaceShape]
MouseActionCursor=1

[TimeAnalogueClockFaceShapeTextSquare]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=Square
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=(#AnalogueClockFaceShape# = 0 ? 0 : 1)

[TimeAnalogueClockFaceShapeTextCircle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=r
Y=r
Text=Circle
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=(#AnalogueClockFaceShape# = 1 ? 0 : 1)

[TimeAnalogueClockFaceShapeTextNone]
Meter=String
MeterStyle=FontStyleVegurRegular
X=r
Y=r
Text=None
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=(#AnalogueClockFaceShape# = 2 ? 0 : 1)

[TimeDateSizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=112
Text=Date size:
StringAlign=Right
FontSize=12

[TimeDateSizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTimeInputText "ExecuteBatch 2"]
MouseActionCursor=1
MouseActionCursorName=Text

[TimeDateSizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#DateSize#
FontSize=12
FontColor=#TertiaryColour#,255

[TimeDateFontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=134
Text=Date font:
StringAlign=Right
FontSize=12

[TimeDateFontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
MouseActionCursor=1
LeftMouseUpAction=[!SetVariable FontToggleVariable DateFont][!SetVariable CurrentFont #DateFont#][!SetVariable FontConfig "Time\Date"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[TimeDateFontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#DateFont#
FontSize=12
FontColor=#TertiaryColour#,255

[TimeDigitalClockSizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=166
Text=Digital clock size:
StringAlign=Right
FontSize=12

[TimeDigitalClockSizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureTimeInputText "ExecuteBatch 3"]
MouseActionCursor=1
MouseActionCursorName=Text

[TimeDigitalClockSizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#DigitalClockSize#
FontSize=12
FontColor=#TertiaryColour#,255

[TimeDigitalClockFontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=188
Text=Digital clock font:
StringAlign=Right
FontSize=12

[TimeDigitalClockFontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable DigitalClockFont][!SetVariable CurrentFont #DigitalClockFont#][!SetVariable FontConfig "Time\Digital Clock"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[TimeDigitalClockFontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#DigitalClockFont#
FontSize=12
FontColor=#TertiaryColour#,255

[TimeDigitalClockLeadingZeroTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=210
Text=Disable leading zero?
StringAlign=Right
FontSize=12

[TimeDigitalClockLeadingZeroBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!EnableMeasure MeasureTimeDigitalClockLeadingZeroToggle]
MouseActionCursor=1

[TimeDigitalClockLeadingZeroTextYes]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=Yes
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=1

[TimeDigitalClockLeadingZeroTextNo]
Meter=String
MeterStyle=FontStyleVegurRegular
X=r
Y=r
Text=No
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=1

[TimeDigitalClockFormatTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=232
Text=Digital clock format:
StringAlign=Right
FontSize=12

[TimeDigitalClockFormatBackground]
Meter=String
X=12r
Y=r
W=100
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!EnableMeasure MeasureTimeDigitalClockFormatToggle]
MouseActionCursor=1

[TimeDigitalClockFormatText24h]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=24 hour clock
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=1

[TimeDigitalClockFormatText12h]
Meter=String
MeterStyle=FontStyleVegurRegular
X=r
Y=r
Text=12 hour clock
FontSize=12
FontColor=#TertiaryColour#,255
Hidden=1
