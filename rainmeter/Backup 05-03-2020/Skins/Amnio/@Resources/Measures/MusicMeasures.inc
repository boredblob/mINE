
;-------------------------------------------
; MEASURES
;-------------------------------------------

[MeasureSong]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=#MusicPlayer#
DisableLeadingZero=1
PlayerType=Title
Substitute="":"Open #MusicPlayerTitle#"

[MeasureCover]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=Cover

[MeasureArtist]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=Artist

[MeasureAlbum]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=Album

[MeasureShuffle]
Measure=Plugin
Plugin=NowPlaying
PlayerName=[MeasureSong]
PlayerType=Shuffle

[MeasureState]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=State
Substitute="0":"Play.png","1":"Pause.png","2":"Play.png"

[MeasureRepeat]
Measure=Plugin
Plugin=NowPlaying
PlayerName=[MeasureSong]
PlayerType=Repeat

[MeasurePosition]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=Position

[MeasureProgress]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=Progress

[MeasureDuration]
Measure=Plugin
Plugin=NowPlaying.dll
PlayerName=[MeasureSong]
PlayerType=Duration

[MeasureDurationLeft]
Measure=Calc
Formula=(MeasureDuration - MeasurePosition)

[MeasureTimeFormat]
Measure=Time
Format=%#M:%S
TimeStamp=[MeasureDurationLeft]
DynamicVariables=1
