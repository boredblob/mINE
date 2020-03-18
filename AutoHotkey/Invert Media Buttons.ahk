﻿#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.

$PrintScreen::Volume_Mute
$ScrollLock::Volume_Down
$Pause::Volume_Up

$Volume_Mute::PrintScreen
$Volume_Down::ScrollLock
$Volume_Up::Pause