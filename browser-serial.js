// Knight Board framing and command timing are documented by NeuroPawn:
// https://docs.neuropawn.tech/knight-board/data-format/
// https://docs.neuropawn.tech/knight-board/command-set/
(function () {
  const SAMPLE_RATE = 125;
  const FRAME_LENGTHS = [21, 57];
  const SYNC_FRAMES = 5;
  const START = 0xA0, END = 0xC0;
  const GAINS = [1, 2, 3, 4, 6, 8, 12];

  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  async function withTimeout(promise, milliseconds, message) {
    let timer;
    try {
      return await Promise.race([promise, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
      })]);
    } finally {
      clearTimeout(timer);
    }
  }

  class Frames {
    constructor() {
      this.pending = new Uint8Array();
      this.length = 0;
    }

    feed(chunk, onFrame) {
      const bytes = new Uint8Array(this.pending.length + chunk.length);
      bytes.set(this.pending);
      bytes.set(chunk, this.pending.length);
      let offset = 0;

      if (!this.length) {
        // Check consecutive counters and boundaries before accepting either format.
        let found = false;
        for (; offset + FRAME_LENGTHS[0] * SYNC_FRAMES <= bytes.length; offset++) {
          if (bytes[offset] !== START) continue;
          const matches = FRAME_LENGTHS.find((length) => {
            if (offset + length * SYNC_FRAMES > bytes.length) return false;
            const counter = bytes[offset + 1];
            for (let i = 0; i < SYNC_FRAMES; i++) {
              const start = offset + i * length;
              if (bytes[start] !== START || bytes[start + length - 1] !== END ||
                  bytes[start + 1] !== ((counter + i) & 255)) return false;
            }
            return true;
          });
          if (matches) {
            this.length = matches;
            found = true;
            break;
          }
        }
        if (!found) {
          this.pending = bytes.slice(-(FRAME_LENGTHS[1] * SYNC_FRAMES - 1));
          return;
        }
      }

      const view = new DataView(bytes.buffer);
      while (offset + this.length <= bytes.length) {
        if (bytes[offset] === START && bytes[offset + this.length - 1] === END) {
          onFrame(view, offset);
          offset += this.length;
        } else {
          offset++;
        }
      }
      this.pending = bytes.slice(offset);
    }
  }

  class Knight {
    constructor(serial, port, { inputs, gain, onDisconnect }) {
      this.fs = SAMPLE_RATE;
      this.inputs = Object.freeze([...inputs]);
      this.gain = gain;
      this.serial = serial;
      this.port = port;
      this.frames = new Frames();
      this.scale = 4e6 / (32767 * 79.57 * gain);
      this.onDisconnect = onDisconnect;
      this.handleDisconnect = (event) => {
        if ((event.port ?? event.target) === port) this.streamLost();
      };
      this.handleVisibility = () => {
        if (document.hidden) this.finishRead(new Error("Keep this tab visible during EEG reads."));
      };
    }

    async open(onStatus) {
      onStatus("Opening NeuroPawn serial port…");
      const opening = this.port.open({ baudRate: 115200, dataBits: 8, stopBits: 1, parity: "none",
        flowControl: "none", bufferSize: 8192 });
      try {
        await withTimeout(opening, 5000, "Opening the serial port timed out.");
      } catch (error) {
        opening.then(() => this.port.close()).catch(() => {});
        if (error.name === "NetworkError") {
          throw new Error("Could not open the NeuroPawn serial port. Close other headset apps and reconnect the board.");
        }
        throw error;
      }
      this.opened = true;

      const started = performance.now();
      this.reader = this.port.readable.getReader();
      this.serial.addEventListener("disconnect", this.handleDisconnect);
      document.addEventListener("visibilitychange", this.handleVisibility);
      let resolveReady, rejectReady;
      const framesReady = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
      this.resolveReady = resolveReady;
      this.rejectReady = rejectReady;
      this.reading = this.readStream();
      onStatus("Checking for Knight Board packets…");
      await withTimeout(framesReady, 6500,
        "No Knight Board data arrived. Select the NeuroPawn serial port and close any other headset apps.");
      this.imu = this.frames.length === 57;

      // Confirm the wire format before sending any configuration to the selected device.
      await delay(Math.max(0, 3000 - (performance.now() - started)));
      const commands = [];
      for (let channel = 1; channel <= 8; channel++) {
        if (!this.inputs.includes(channel)) commands.push(`rldremove_${channel}`, `choff_${channel}`);
      }
      for (const channel of this.inputs) commands.push(`chon_${channel}_${this.gain}`, `rldadd_${channel}`);
      const encoder = new TextEncoder();
      for (const [index, command] of commands.entries()) {
        if (this.closing) throw new Error("The NeuroPawn disconnected during setup.");
        onStatus(`Configuring NeuroPawn (${index + 1}/${commands.length})…`);
        const writer = this.port.writable.getWriter();
        // BrainFlow sends the command bytes verbatim, with no line terminator.
        this.writing = withTimeout(writer.write(encoder.encode(command)), 2500,
          "The NeuroPawn stopped accepting commands.").finally(() => writer.releaseLock());
        await this.writing;
        await delay(1500);
      }
      if (this.closing) throw new Error("The NeuroPawn disconnected during setup.");
      if (performance.now() - this.lastFrame > 2500) {
        throw new Error("The NeuroPawn stopped sending data during setup. Reconnect it and try again.");
      }
      this.ready = true;
    }

    async readStream() {
      const reader = this.reader;
      try {
        while (!this.closing) {
          const { value, done } = await reader.read();
          if (done) break;
          this.frames.feed(value, (view, offset) => this.onFrame(view, offset));
        }
      } catch {
        // Port unplug, buffer overrun, and hardware read errors all invalidate the current read.
      } finally {
        reader.releaseLock();
        this.reader = null;
        if (!this.closing) this.streamLost();
      }
    }

    streamLost() {
      if (this.closing) return;
      const notify = this.ready;
      const error = new Error("The NeuroPawn serial stream stopped. Reconnect the board and try again.");
      this.rejectReady?.(error);
      this.close(error).catch(() => {});
      if (notify) this.onDisconnect(error);
    }

    onFrame(view, offset) {
      this.lastFrame = performance.now();
      this.resolveReady?.();
      this.resolveReady = this.rejectReady = null;
      const read = this.read;
      if (!read) return;
      const counter = view.getUint8(offset + 1);
      if (read.counter !== null && counter !== ((read.counter + 1) & 255)) {
        this.finishRead(new Error("EEG packets were dropped. Keep the headset connected and retry this read."));
        return;
      }
      read.counter = counter;
      for (const [row, channel] of this.inputs.entries()) {
        const count = view.getInt16(offset + 2 * channel);
        if (count === 32767 || count === -32768) {
          this.finishRead(new Error(`EEG clipped on input ${channel}. Check the electrode contacts and ear clip, then retry.`));
          return;
        }
        read.samples[row][read.length] = count * this.scale;
        if (read.length === 0) read.first[row] = count;
        else if (count !== read.first[row]) read.varied[row] = true;
      }
      if (++read.length === read.samples[0].length) {
        const flat = read.varied.indexOf(false);
        this.finishRead(flat === -1 ? null : new Error(
          `No changing signal on input ${this.inputs[flat]}. Check its electrode and retry the read.`));
      }
    }

    capture(seconds, { onStart } = {}) {
      const sampleCount = Math.round(this.fs * seconds);
      if (!Number.isFinite(seconds) || sampleCount < 1 || seconds > 30) {
        return Promise.reject(new Error("Use a positive EEG read duration of at most 30 seconds."));
      }
      if (!this.ready || this.closing) return Promise.reject(new Error("Connect the NeuroPawn before recording."));
      if (this.read) return Promise.reject(new Error("Another EEG read is in progress."));
      if (document.hidden) return Promise.reject(new Error("Keep this tab visible during EEG reads."));
      return new Promise((resolve, reject) => {
        this.read = {
          resolve, reject, length: 0, counter: null,
          samples: this.inputs.map(() => new Float32Array(sampleCount)),
          first: this.inputs.map(() => 0), varied: this.inputs.map(() => sampleCount === 1),
          timer: setTimeout(() => this.finishRead(new Error("Timed out waiting for EEG samples. Recheck the USB connection.")),
            seconds * 1000 + 3000)
        };
        try { onStart?.(); } catch (error) { this.finishRead(error); }
      });
    }

    finishRead(error) {
      const read = this.read;
      if (!read) return;
      this.read = null;
      clearTimeout(read.timer);
      if (error) read.reject(error);
      else read.resolve(read.samples);
    }

    async close(error = new Error("The NeuroPawn was disconnected.")) {
      if (this.closing) return this.closing;
      this.ready = false;
      this.finishRead(error);
      this.serial.removeEventListener("disconnect", this.handleDisconnect);
      document.removeEventListener("visibilitychange", this.handleVisibility);
      this.closing = (async () => {
        if (!this.opened) return;
        await this.reader?.cancel().catch(() => {});
        await this.reading;
        await this.writing?.catch(() => {});
        await withTimeout(this.port.close(), 3000, "The serial port did not close. Unplug and reconnect the board.");
        this.opened = false;
      })();
      return this.closing;
    }
  }

  async function connect({ inputs = [1, 2, 6], gain = 4, onStatus = () => {}, onDisconnect = () => {} } = {}) {
    const serial = navigator.serial;
    if (!window.isSecureContext || typeof serial?.requestPort !== "function") {
      throw new Error("Use this HTTPS page in desktop Chrome or Edge to connect the NeuroPawn.");
    }
    if (!Array.isArray(inputs) || !inputs.length || new Set(inputs).size !== inputs.length ||
        !inputs.every((input) => Number.isInteger(input) && input >= 1 && input <= 8) || !GAINS.includes(gain)) {
      throw new Error("Choose unique NeuroPawn inputs from 1 to 8 and a supported gain.");
    }
    let port;
    try {
      port = await serial.requestPort();
    } catch (error) {
      if (error.name === "NotFoundError") throw new Error("No serial port was selected. Choose the NeuroPawn when connecting.");
      throw error;
    }
    const board = new Knight(serial, port, { inputs, gain, onDisconnect });
    try {
      await board.open(onStatus);
      return board;
    } catch (error) {
      await board.close().catch(() => {});
      throw error;
    }
  }

  window.NPSerial = { connect };
})();
