# MedVision

MedVision is a desktop assistant for visually impaired users. It detects nearby objects and traffic lights from a laptop or external webcam and speaks short Uzbek navigation alerts offline.

## Quick Start

Double-click:

```text
install_and_start.bat
```

For later launches, double-click:

```text
start.bat
```

## Manual Run

```powershell
cd C:\Ai\MedVision
python -m pip install -r requirements.txt
python main.py
```

## Camera Selection

Open the app, choose a camera from the list, then press `Boshlash`.

If you connect a USB webcam after opening the app, press `Kameralarni yangilash`.

## Console Mode

```powershell
python main.py --console
```
