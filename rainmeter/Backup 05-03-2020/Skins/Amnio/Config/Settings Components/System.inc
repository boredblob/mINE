
[SystemArrow]
Meter=Image
ImageName=#@#Images\SettingsArrow.png
X=36
Y=96
ImageTint=#MainColour#,150

[SystemTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=48
Text=System
FontSize=30

[SystemTitleLine]
Meter=String
X=42
Y=50
W=400
H=1
SolidColor=#MainColour#,150

[SystemCPUSizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=58
Text=CPU size:
StringAlign=Right
FontSize=12

[SystemCPUSizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 1"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemCPUSizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#CPUSize#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemCPUFontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=80
Text=CPU font:
StringAlign=Right
FontSize=12

[SystemCPUFontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable CPUFont][!SetVariable CurrentFont #CPUFont#][!SetVariable FontConfig "System\CPU"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[SystemCPUFontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#CPUFont#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemPowerSizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=112
Text=Power size:
StringAlign=Right
FontSize=12

[SystemPowerSizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 2"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemPowerSizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#PowerSize#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemPowerFontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=134
Text=Power font:
StringAlign=Right
FontSize=12

[SystemPowerFontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable PowerFont][!SetVariable CurrentFont #PowerFont#][!SetVariable FontConfig "System\Power"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[SystemPowerFontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#PowerFont#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemRAMSizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=166
Text=RAM size:
StringAlign=Right
FontSize=12

[SystemRAMSizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 3"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemRAMSizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#RAMSize#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemRAMFontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=188
Text=RAM font:
StringAlign=Right
FontSize=12

[SystemRAMFontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable RAMFont][!SetVariable CurrentFont #RAMFont#][!SetVariable FontConfig "System\RAM"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[SystemRAMFontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#RAMFont#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD1SizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=220
Text=1st HDD size:
StringAlign=Right
FontSize=12

[SystemHDD1SizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 4"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemHDD1SizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD1Size#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD1DriveTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=242
Text=1st HDD drive:
StringAlign=Right
FontSize=12

[SystemHDD1DriveBackground]
Meter=String
X=12r
Y=r
W=20
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 5"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemHDD1DriveText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD1Drive#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD1FontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=264
Text=1st HDD font:
StringAlign=Right
FontSize=12

[SystemHDD1FontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable HDD1Font][!SetVariable CurrentFont #HDD1Font#][!SetVariable FontConfig "System\HDD 1"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[SystemHDD1FontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD1Font#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD2SizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=296
Text=2nd HDD size:
StringAlign=Right
FontSize=12

[SystemHDD2SizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 6"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemHDD2SizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD2Size#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD2DriveTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=318
Text=2nd HDD drive:
StringAlign=Right
FontSize=12

[SystemHDD2DriveBackground]
Meter=String
X=12r
Y=r
W=20
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 7"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemHDD2DriveText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD2Drive#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD2FontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=340
Text=2nd HDD font:
StringAlign=Right
FontSize=12

[SystemHDD2FontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable HDD2Font][!SetVariable CurrentFont #HDD2Font#][!SetVariable FontConfig "System\HDD 2"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[SystemHDD2FontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD2Font#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD3SizeTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=372
Text=3rd HDD size:
StringAlign=Right
FontSize=12

[SystemHDD3SizeBackground]
Meter=String
X=12r
Y=r
W=40
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 8"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemHDD3SizeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD3Size#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD3DriveTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=394
Text=3rd HDD drive:
StringAlign=Right
FontSize=12

[SystemHDD3DriveBackground]
Meter=String
X=12r
Y=r
W=20
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!CommandMeasure MeasureSystemInputText "ExecuteBatch 9"]
MouseActionCursor=1
MouseActionCursorName=Text

[SystemHDD3DriveText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD3Drive#
FontSize=12
FontColor=#TertiaryColour#,255

[SystemHDD3FontTitle]
Meter=String
MeterStyle=FontStyleVegurRegular
X=236
Y=416
Text=3rd HDD font:
StringAlign=Right
FontSize=12

[SystemHDD3FontBackground]
Meter=String
X=12r
Y=r
W=80
H=18
SolidColor=#MainColour#,150
LeftMouseUpAction=[!SetVariable FontToggleVariable HDD3Font][!SetVariable CurrentFont #HDD3Font#][!SetVariable FontConfig "System\HDD 3"][!EnableMeasure MeasureFontToggle]
MouseActionCursor=1

[SystemHDD3FontText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=2r
Y=r
Text=#HDD3Font#
FontSize=12
FontColor=#TertiaryColour#,255
