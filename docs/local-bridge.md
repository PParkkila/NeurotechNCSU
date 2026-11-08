# Run Shape Battle with a NeuroPawn

Run the bridge on the laptop that has the NeuroPawn connected over USB. Extract this ZIP, then open a terminal in the extracted `neuropawn-local` folder. Use Python 3.11–3.14.

## macOS

Plug in the board and use `ls /dev/cu.usb*` to find its serial port. Close EXG Visualizer if it is using the port. Replace the example port in the last command with the one on your laptop.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m imagery web --port /dev/cu.usbserial1234
```

## Windows

Find the board's COM port under **Ports (COM & LPT)** in Device Manager. Close EXG Visualizer if it is using the port. Replace `COM3` with the port on your laptop.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m imagery web --port COM3
```

## Start calibration

Keep the terminal open. The board takes about 30 seconds to configure. Once the terminal prints **Open http://127.0.0.1:8765**, open that address in a browser on the same laptop and click **Start calibration**. Complete 20 reads to unlock play against the computer.

If the address fails to open, check the terminal for an error and confirm the board is connected on the selected port. If port 8765 is already in use, append `--listen-port 8766` to the bridge command and use the address it prints.

The game defaults to gain 4 and board inputs 1, 2, and 6. Pass `--inputs` and `--gain` to match your wiring. EEG session files are saved locally under `runs/web-sessions/`.
