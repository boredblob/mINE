
;-------------------------------------------
; MEASURES
;-------------------------------------------
; TIME

[MeasureDigitalTime]
Measure=Time
Format=%#DigitalClockLeadingZero##DigitalClockFormat#:%M

[MeasureDate]
Measure=Time
Format=%A, %#d %B

;-------------------------------------------
; POWER

[MeasurePower]
Measure=Plugin
Plugin=PowerPlugin.dll
PowerState=Percent
UpdateDivider=60
IfCondition=MeasurePower > -1
IfTrueAction=[!SetVariable PowerState 25]
IfCondition2=MeasurePower > 25
IfTrueAction2=[!SetVariable PowerState 50]
IfCondition3=MeasurePower > 50
IfTrueAction3=[!SetVariable PowerState 75]
IfCondition4=MeasurePower > 75
IfTrueAction4=[!SetVariable PowerState 100]

[MeasurePowerACLine]
Measure=Plugin
Plugin=PowerPlugin.dll
PowerState=ACLine
UpdateDivider=60
Substitute="0":"Not charging","1":"Charging"

[MeasurePowerUptime]
Measure=Uptime
Format="%3!i!h, %2!i!m"
AddDaysToHours=1
UpdateDivider=60

;-------------------------------------------
; VOLUME

[MeasureVolume]
Measure=Plugin
Plugin=Win7AudioPlugin.dll

[MeasureSetVolume]
Measure=Calc
Formula=#VolumePercentage#
DynamicVariables=1
OnChangeAction=[!CommandMeasure MeasureVolume "SetVolume #VolumePercentage#"][!Update]

[MeasureVolumeIcon]
Measure=Calc
Formula=MeasureVolume
IfCondition=MeasureVolumeIcon < 0
IfTrueAction=[!SetVariable VolumeState -1][!UpdateMeter VolumeExpand][!Redraw]
IfCondition2=MeasureVolumeIcon = 0
IfTrueAction2=[!SetVariable VolumeState 0][!UpdateMeter VolumeExpand][!Redraw]
IfCondition3=(MeasureVolumeIcon > 0) && (MeasureVolumeIcon < 33)
IfTrueAction3=[!SetVariable VolumeState 1][!UpdateMeter VolumeExpand][!Redraw]
IfCondition4=MeasureVolumeIcon > 32 && (MeasureVolumeIcon < 66)
IfTrueAction4=[!SetVariable VolumeState 2][!UpdateMeter VolumeExpand][!Redraw]
IfCondition5=MeasureVolumeIcon > 65
IfTrueAction5=[!SetVariable VolumeState 3][!UpdateMeter VolumeExpand][!Redraw]

[MeasureVolumePercent]
Measure=Calc
Formula=MeasureVolume
Substitute="-1":"Muted"
IfCondition=MeasureVolumePercent > -1
IfTrueAction=[!SetOption VolumePercentage Postfix %]
IfFalseAction=[!SetOption VolumePercentage Postfix ""]

;-------------------------------------------
; SYSTEM

[MeasureCPU]
Measure=CPU

[MeasureRAM]
Measure=PhysicalMemory
UpdateDivider=3

[MeasureRAMFree]
Measure=PhysicalMemory
InvertMeasure=1
UpdateDivider=3

;-------------------------------------------
; BIN

[MeasureBinCount]
Measure=Plugin
Plugin=RecycleManager
RecycleType=Count
UpdateDivider=3
IfCondition=MeasureBinCount > 0
IfTrueAction=[!SetVariable BinState Full]
IfFalseAction=[!SetVariable BinState Empty]

[MeasureBinSize]
Measure=Plugin
Plugin=RecycleManager
RecycleType=Size
UpdateDivider=3
