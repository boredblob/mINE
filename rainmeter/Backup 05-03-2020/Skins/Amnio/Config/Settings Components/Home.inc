
[HomeWelcomeToTitle]
Meter=String
MeterStyle=FontStyleVegurLight
X=242
Text=Welcome To
StringAlign=Center
FontSize=12

[Home]
Meter=String
MeterStyle=FontStyleVegurLight
X=242
Y=8
Text=Amnio
StringAlign=Center
FontSize=40

[HomeEnglishBackground]
Meter=String
X=102
Y=85
W=80
H=30
SolidColor=#SecondaryColour#,15

[HomeEnglish]
Meter=String
MeterStyle=FontStyleVegurLight
X=40r
Y=15r
W=80
H=30
Text=English
StringAlign=CenterCenter
FontSize=16
SolidColor=#SecondaryColour#,1
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=[!WriteKeyValue Variables Analogue ue "#@#Variables.inc"][!WriteKeyValue Variables Colour u "#@#Variables.inc"]
MouseActionCursor=1

[HomeAmericanEnglishBackground]
Meter=String
X=222
Y=-15r
W=160
H=30
SolidColor=#SecondaryColour#,15

[HomeAmericanEnglish]
Meter=String
MeterStyle=FontStyleVegurLight
X=80r
Y=15r
W=160
H=30
Text=American English
StringAlign=CenterCenter
FontSize=16
SolidColor=#SecondaryColour#,1
MouseOverAction=[!SetOption #CurrentSection# SolidColor #SecondaryColour#,15][!UpdateMeter #CurrentSection#][!Redraw]
MouseLeaveAction=[!SetOption #CurrentSection# SolidColor 0,0,0,1][!UpdateMeter #CurrentSection#][!Redraw]
LeftMouseUpAction=[!WriteKeyValue Variables Analogue "" "#@#Variables.inc"][!WriteKeyValue Variables Colour "" "#@#Variables.inc"]
MouseActionCursor=1

[HomeText]
Meter=String
MeterStyle=FontStyleVegurRegular
X=46
Y=142
W=396
H=110
Text=This suite was created by JosephB2000 and is inspired by viclarsson's 'Rainmeter Clean Skin'.#CRLF##CRLF#This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
FontSize=12
ClipString=2
