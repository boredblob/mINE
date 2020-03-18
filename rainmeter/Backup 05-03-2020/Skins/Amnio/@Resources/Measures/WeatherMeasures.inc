
;---------------------------------------------
; MAIN MEASURES
;---------------------------------------------

[MeasureCurrent]
Measure=Plugin
Plugin=Plugins\WebParser.dll
UpdateRate=1800
Url=http://xml.weather.com/weather/local/#WeatherCode#?cc=*&unit=#Unit#&dayf=0
RegExp="(?siU).*<tmp>(.*)</tmp>.*<t>(.*)</t>.*<icon>(.*)</icon>"

[MeasureTomorrow]
Measure=Plugin
Plugin=Plugins\WebParser.dll
UpdateRate=1800
Url=http://xml.weather.com/weather/local/#WeatherCode#?cc=*&unit=#Unit#&dayf=2
RegExp="(?siU)<day d="1" t="(.*)".*<hi>(.*)</hi>.*<low>(.*)</low>.*<icon>(.*)</icon>"

[MeasureDayAfter]
Measure=Plugin
Plugin=Plugins\WebParser.dll
UpdateRate=1800
Url=http://xml.weather.com/weather/local/#WeatherCode#?cc=*&unit=#Unit#&dayf=3
RegExp="(?siU)<day d="2" t="(.*)".*<hi>(.*)</hi>.*<low>(.*)</low>.*<icon>(.*)</icon>"

[MeasureDayAfter2]
Measure=Plugin
Plugin=Plugins\WebParser.dll
UpdateRate=1800
Url=http://xml.weather.com/weather/local/#WeatherCode#?cc=*&unit=#Unit#&dayf=4
RegExp="(?siU)<day d="3" t="(.*)".*<hi>(.*)</hi>.*<low>(.*)</low>.*<icon>(.*)</icon>"

;---------------------------------------------
; CURRENT
;---------------------------------------------

[MeasureWeatherTemp]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureCurrent]
StringIndex=1

[MeasureWeatherDesc]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureCurrent]
StringIndex=2
Substitute=" And ":" and ","Snow/":"Snow and ","Lgt.":"Light ","T-Storm":"Thunder Storm","":"Unknown"

[MeasureWeatherIcon]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureCurrent]
StringIndex=3
Substitute="":"na"

;---------------------------------------------
; TOMORROW
;---------------------------------------------

[MeasureWeatherDay1]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureTomorrow]
StringIndex=1
Substitute="":"Unknown"

[MeasureWeatherTempHigh1]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureTomorrow]
StringIndex=2

[MeasureWeatherTempLow1]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureTomorrow]
StringIndex=3

[MeasureWeatherIcon1]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureTomorrow]
StringIndex=4
Substitute="":"na"

;---------------------------------------------
; DAY AFTER
;---------------------------------------------

[MeasureWeatherDay2]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter]
StringIndex=1
Substitute="":"Unknown"

[MeasureWeatherTempHigh2]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter]
StringIndex=2

[MeasureWeatherTempLow2]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter]
StringIndex=3

[MeasureWeatherIcon2]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter]
StringIndex=4
Substitute="":"na"

;---------------------------------------------
; DAY AFTER 2
;---------------------------------------------

[MeasureWeatherDay3]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter2]
StringIndex=1
Substitute="":"Unknown"

[MeasureWeatherTempHigh3]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter2]
StringIndex=2

[MeasureWeatherTempLow3]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter2]
StringIndex=3

[MeasureWeatherIcon3]
Measure=Plugin
Plugin=Plugins\WebParser.dll
Url=[MeasureDayAfter2]
StringIndex=4
Substitute="":"na"
